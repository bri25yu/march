from torch import BoolTensor, LongTensor, cat, full, masked_select
from torch.nn.functional import dropout, embedding, softmax

from march.models.baseline import *
from march.models.utils import *
from march.models.utils import Tuple


__all__ = ["FCABSTransformer", "FCABSTransformerConfig"]


# Fixed change attention based summarization (FCABS)
# Fixed change meaning there is a fixed number of tokens to drop per layer
# Attention based means we use attention activations to determine which tokens to drop
# Summarization means we drop tokens from the sequence length


@dataclass
class FCABSAttentionOutput:
    input_embeds: SequenceInputEmbeds
    position_bias: MultiHeadedAttention
    mask_drop: SequenceInputIds
    bottomk_indices: SequenceInputIds
    effective_L_drop: int


@dataclass
class FCABSEncoderOutput:
    input_embeds: SequenceInputEmbeds
    position_bias: MultiHeadedAttention
    attention_mask: SequenceInputIds
    mask_drop_by_layer: List[SequenceInputIds] = None
    dropped_indices_by_layer: List[SequenceInputIds] = None
    effective_L_drop_by_layer: List[int] = None


@dataclass
class FCABSTransformerOutput(Seq2SeqLMOutput):
    dropped_ids: SequenceInputIds=None


def update_L_dimension(
    input_embeds: FloatTensor,
    attention_mask: BoolTensor,
    position_bias: FloatTensor,
    mask_drop: BoolTensor,
    effective_L_drop: int,
) -> Tuple[FloatTensor, BoolTensor, FloatTensor]:
    """
    input_embeds is (N, L, D)
    attention_mask is (N, L)
    position_bias is (N, H, L, L)
    mask_drop is (N, L)
    """
    N, L, D = input_embeds.size()
    _, H, _, _ = position_bias.size()
    L_out = L - effective_L_drop

    mask_keep = ~mask_drop

    # Drop from input_embeds
    expanded_mask_keep = mask_keep[:, :, None].repeat(1, 1, D)
    assert expanded_mask_keep.size() == (N, L, D)
    input_embeds = masked_select(input_embeds, expanded_mask_keep).reshape(N, L_out, D)

    # Drop from attention_mask
    attention_mask = masked_select(attention_mask, mask_keep).reshape(N, L_out)

    # Drop from position_bias
    expanded_mask_keep = mask_keep[:, None, :].repeat(1, H, 1)  # (N, H, L)
    expanded_mask_keep = expanded_mask_keep[:, :, None, :] & expanded_mask_keep[:, :, :, None]
    assert expanded_mask_keep.size() == (N, H, L, L)
    position_bias = masked_select(position_bias, expanded_mask_keep).reshape(N, H, L_out, L_out)

    return input_embeds, attention_mask, position_bias


def calculate_mask_drop(attention_probs: FloatTensor, L_drop: int) -> Tuple[BoolTensor, LongTensor, int]:
    """
    attention_probs is (N, H, L, L)
    """
    N, H, L, L = attention_probs.size()
    effective_L_drop = 0 if L_drop > (L // 2) else L_drop

    attention_probs_for_mask = attention_probs.view(N, H * L, L).sum(dim=1)
    assert attention_probs_for_mask.size() == (N, L)

    bottomk_indices = attention_probs_for_mask.topk(k=effective_L_drop, dim=1, largest=False).indices

    mask_drop = full((N, L), False, dtype=bool, device=attention_probs.device)
    mask_drop.scatter_(dim=1, index=bottomk_indices, value=True)

    return mask_drop, bottomk_indices, effective_L_drop


def log_fcabs(
    input_ids: SequenceInputIds,
    dropped_indices_by_layer: List[SequenceInputIds],
    mask_drop_by_layer: List[SequenceInputIds],
    effective_L_drop_by_layer: List[int],
):
    """
    N_L is the number of layers

    input_ids is (N, L)
    dropped_indices_by_layer is (N_L, N, L_drop)
    mask_drop_by_layer is (N_L, N, L)
    effective_L_drop_by_layer is length N_L

    Returns dropped_ids of shape (N, N_L, L_drop)
    """
    if dropped_indices_by_layer is None: return None

    N, L_out = input_ids.size()

    dropped_ids_by_layer = []  # (N_L, N, L_drop)
    for dropped_indices, mask_drop, effective_L_drop in zip(dropped_indices_by_layer, mask_drop_by_layer, effective_L_drop_by_layer):
        L_out -= effective_L_drop

        dropped_ids = input_ids.gather(dim=1, index=dropped_indices)
        assert dropped_ids.size() == (N, effective_L_drop)
        input_ids = masked_select(input_ids, ~mask_drop).reshape(N, L_out)

        dropped_ids_by_layer.append(dropped_ids[None, :, :])  # Expand for cat later

    dropped_ids = cat(dropped_ids_by_layer, dim=0).permute(1, 0, 2)

    return dropped_ids


@dataclass
class FCABSTransformerConfig(TransformerConfig):
    # Per-layer drop in the sequence dimension
    L_drop: int = 8


class FCABSAttention(BaselineAttention):
    def forward(
        self,
        input_embeds: SequenceInputEmbeds,
        attention_mask: SequenceInputIds=None,
        position_bias: MultiHeadedAttention=None,
        encoder_hidden_state: SequenceInputEmbeds=None,
    ) -> AttentionOutput:
        config = self.config

        key_value_state = encoder_hidden_state if self.is_cross_attention else input_embeds
        query, key, value = list(map(
            self.reshape_to_head_sensitive,
            [self.w_q(input_embeds), self.w_k(key_value_state), self.w_v(key_value_state)]
        ))

        attention_logits: MultiHeadedAttention = matmul(query, key.transpose(2, 3))

        if position_bias is None:
            batch_size, _, query_length, key_length = attention_logits.size()

            position_bias = self.compute_bias(query_length, key_length, attention_logits.device, attention_logits.dtype).repeat(batch_size, 1, 1, 1)

            # Convert attention mask to mask to add to attention logits
            attention_mask = attention_mask.reshape(batch_size, 1, -1, key_length)
            attention_mask = attention_mask.to(attention_logits.dtype) * finfo(attention_logits.dtype).min

            # Combine position bias and attention masks to save on computation in subsequent layers
            # This saves (2L - 2) * (N * H * L * L) additions per model pass
            position_bias = position_bias + attention_mask

        attention_logits: MultiHeadedAttention = attention_logits + position_bias
        attention_probs: MultiHeadedAttention = softmax(attention_logits.to(float32), dim=3).to(attention_logits.dtype)

        # Use attention probs output of softmax to calculate which tokens to drop, 	
        # based on which has the lowest attention activation.	
        # Attention_probs is of dimension (N, H, L_q, L_k). Sum over H and L_q dimensions	
        mask_drop, bottomk_indices, effective_L_drop = calculate_mask_drop(attention_probs, config.L_drop) # (N, L)	

        attention_probs: MultiHeadedAttention = dropout(attention_probs, p=config.dropout_prob, training=self.training)
        attention_values: SequenceInputEmbeds = self.reshape_to_head_insensitive(matmul(attention_probs, value))
        attention_output: SequenceInputEmbeds = self.w_o(attention_values)
        return FCABSAttentionOutput(
            input_embeds=attention_output,
            position_bias=position_bias,
            mask_drop=mask_drop,
            bottomk_indices=bottomk_indices,
            effective_L_drop=effective_L_drop,
        )


class FCABSEncoder(BaselineEncoder):
    ATTENTION_CLS = FCABSAttention

    def forward(self, input_embeds: SequenceInputEmbeds, attention_mask: SequenceInputIds, return_fcabs: bool=False) -> AttentionOutput:
        selfattn_ln: LayerNorm
        selfattn: AttentionBase
        ff_ln: LayerNorm
        ff: TransformerComponentBase

        input_embeds = self.apply_dropout(input_embeds)
        position_bias = None
        if return_fcabs:
            mask_drop_by_layer, dropped_indices_by_layer, effective_L_drop_by_layer = [], [], []
        else:
            mask_drop_by_layer, dropped_indices_by_layer, effective_L_drop_by_layer = None, None, None

        for selfattn_ln, selfattn, ff_ln, ff in self.layers:
            # Also take the mask_drop here
            self_attention_output: FCABSAttentionOutput = selfattn(selfattn_ln(input_embeds), attention_mask, position_bias)
            input_embeds = self.apply_residual(input_embeds, self_attention_output.input_embeds)
            position_bias = self_attention_output.position_bias

            # Update L dimension by dropping the indices which had the lowest attention activation, calculated earlier in self attention
            input_embeds, attention_mask, position_bias = update_L_dimension(
                input_embeds, attention_mask, position_bias, self_attention_output.mask_drop, self_attention_output.effective_L_drop
            )
            if return_fcabs:
                mask_drop_by_layer.append(self_attention_output.mask_drop)
                dropped_indices_by_layer.append(self_attention_output.bottomk_indices)
                effective_L_drop_by_layer.append(self_attention_output.effective_L_drop)

            input_embeds = self.apply_residual(input_embeds, ff(ff_ln(input_embeds)))

        input_embeds = self.apply_dropout(self.final_layernorm(input_embeds))

        return FCABSEncoderOutput(
            input_embeds=input_embeds,
            position_bias=None,
            attention_mask=attention_mask,
            mask_drop_by_layer=mask_drop_by_layer,
            dropped_indices_by_layer=dropped_indices_by_layer,
            effective_L_drop_by_layer=effective_L_drop_by_layer,
        )


class FCABSTransformer(BaselineTransformer):
    ENCODER_CLS = FCABSEncoder

    def forward(
        self,
        input_ids: SequenceInputIds,
        attention_mask: SequenceInputIds,
        decoder_input_ids: SequenceInputIds,
        decoder_attention_mask: SequenceInputIds,
        labels: SequenceInputIds,
    ) -> Seq2SeqLMOutput:
        config = self.config

        # Should log fcabs?
        return_fcabs = not self.training

        input_embeds: SequenceInputEmbeds = embedding(input_ids, self.embedding.weight)
        encoder_outputs: FCABSEncoderOutput = self.encoder(input_embeds, attention_mask, return_fcabs=return_fcabs)
        encoder_hidden_state = encoder_outputs.input_embeds

        decoder_input_embeds: SequenceInputEmbeds = embedding(decoder_input_ids, self.embedding.weight)
        # Use updated attention mask from FCABS encoder
        decoder_outputs: AttentionOutput = self.decoder(
            decoder_input_embeds, decoder_attention_mask, encoder_hidden_state, encoder_outputs.attention_mask,
        )

        sequence_output = decoder_outputs.input_embeds
        sequence_output = sequence_output * (config.dim_model ** -0.5)

        lm_logits = self.embedding(sequence_output)

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        dropped_ids = log_fcabs(
            input_ids=input_ids,
            dropped_indices_by_layer=encoder_outputs.dropped_indices_by_layer,
            mask_drop_by_layer=encoder_outputs.mask_drop_by_layer,
            effective_L_drop_by_layer=encoder_outputs.effective_L_drop_by_layer,
        )

        return FCABSTransformerOutput(loss=loss, logits=lm_logits, dropped_ids=dropped_ids)
