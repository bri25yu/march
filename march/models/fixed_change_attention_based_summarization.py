from typing import List, Type

from march.models.baseline import *
from march.models.utils import *

from torch.nn.functional import dropout, embedding, relu, softmax

from torch import BoolTensor, masked_select

__all__ = ["FCABSTransformer", "FCABSTransformerConfig"]


@dataclass
class FCABSAttentionOutput:
    input_embeds: SequenceInputEmbeds
    position_bias: MultiHeadedAttention
    mask_drop: SequenceInputIds


@dataclass
class FCABSEncoderOutput:
    input_embeds: SequenceInputEmbeds
    position_bias: MultiHeadedAttention
    attention_mask: SequenceInputIds


def reduce_input_embeds_L_dim(input_embeds: FloatTensor, mask_drop: BoolTensor) -> FloatTensor:
    """
    input_embeds: FloatTensor of shape (N, L, D)
    mask_drop: BoolTensor of shape (N, L) where mask.sum(dim=1) = effective_L_drop
        Ones indicate masking and zeros indicate keep
    """
    N, L, D = input_embeds.size()

    expanded_mask_drop = mask_drop[:, :, None].repeat(1, 1, D)
    assert expanded_mask_drop.size() == (N, L, D)

    reduced_L_embeds = masked_select(input_embeds, ~expanded_mask_drop).reshape(N, -1, D)

    return reduced_L_embeds


def reduce_position_bias_L_dim(position_bias: FloatTensor, mask_drop: BoolTensor) -> FloatTensor:
    """
    position_bias: FloatTensor of shape (N, H, L, L)
    mask_drop: BoolTensor of shape (N, L) where mask.sum(dim=1) = effective_L_drop
        Ones indicate masking and zeros indicate keep
    """
    N, H, L, L = position_bias.size()

    L_drop = mask_drop.sum(dim=1)[0]
    L_out = L - L_drop

    expanded_mask_drop = mask_drop[:, None, :].repeat(1, H, 1)
    assert expanded_mask_drop.size() == (N, H, L)

    # Repeat the mask to duplicate the L_out dimension 
    expanded_mask_drop = expanded_mask_drop[:, :, None, :] | expanded_mask_drop[:, :, :, None]
    assert expanded_mask_drop.size() == (N, H, L, L)

    reduced_L_position_bias = masked_select(position_bias, ~expanded_mask_drop).reshape(N, H, L_out, L_out)
    return reduced_L_position_bias


def update_L_dimension(
    input_embeds: FloatTensor, attention_mask: BoolTensor, position_bias: FloatTensor, mask_drop: BoolTensor
) -> Tuple[FloatTensor, BoolTensor, FloatTensor]:
    """
    input_embeds is (N, L, D)
    attention_mask is (N, L)
    position_bias is (1, L, L)
    mask_drop is (N, L)
    """
    input_embeds = reduce_input_embeds_L_dim(input_embeds, mask_drop)
    N = input_embeds.size()[0]
    attention_mask = masked_select(attention_mask, ~mask_drop).reshape(N, -1)
    position_bias = reduce_position_bias_L_dim(position_bias, mask_drop)

    return input_embeds, attention_mask, position_bias


def calculate_mask_drop(attention_probs: FloatTensor, L_drop: int) -> BoolTensor:
    """
    attention_probs is (N, H, L, L)
    """
    N, H, L, L = attention_probs.size()
    effective_L_drop = 0 if L_drop > (L // 2) else L_drop

    attention_probs_for_mask = attention_probs.view(N, H * L, L).sum(dim=1)
    assert attention_probs_for_mask.size() == (N, L)

    bottomk_indices = attention_probs_for_mask.topk(k=effective_L_drop, dim=1, largest=False).indices

    mask_drop = BoolTensor(N, L, device=attention_probs.device)
    mask_drop.scatter_(dim=1, index=bottomk_indices, value=True)

    return mask_drop


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

        if not self.is_cross_attention:
            attention_values: List[SequenceInputEmbeds] = self.w_q(input_embeds), self.w_k(input_embeds), self.w_v(input_embeds)
        else:
            query: SequenceInputEmbeds = self.w_q(input_embeds)
            key: SequenceInputEmbeds = self.w_k(encoder_hidden_state)
            value: SequenceInputEmbeds = self.w_v(encoder_hidden_state)

            attention_values: List[SequenceInputEmbeds] = (query, key, value)

        query, key, value = list(map(self.reshape_to_head_sensitive, attention_values))

        attention_logits: MultiHeadedAttention = matmul(query, key.transpose(2, 3))

        # Infer is_decoder from attention mask size
        is_decoder = len(attention_mask.size()) == 3

        batch_size, _, query_length, key_length = attention_logits.size()
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(batch_size, 1, -1, key_length)
            attention_mask = attention_mask.to(attention_logits.dtype) * finfo(attention_logits.dtype).min
            attention_logits: MultiHeadedAttention = attention_logits + attention_mask

        if position_bias is not None:
            attention_logits: MultiHeadedAttention = attention_logits + position_bias
        elif self.has_relative_attention_bias:
            # Repeat batch size here since they will be unique per example after the first sequence length summarization
            position_bias = self.compute_bias(query_length, key_length, is_decoder).repeat(batch_size, 1, 1, 1)
            attention_logits: MultiHeadedAttention = attention_logits + position_bias

        attention_probs: MultiHeadedAttention = softmax(attention_logits.to(float32), dim=3).to(attention_logits.dtype)

        # Use attention probs output of softmax to calculate which tokens to drop, 
        # based on which has the lowest attention activation.
        # Attention_probs is of dimension (N, H, L_q, L_k). Sum over H and L_q dimensions
        mask_drop = calculate_mask_drop(attention_probs, config.L_drop) # (N, L)
        
        attention_probs: MultiHeadedAttention = dropout(attention_probs, p=config.dropout_prob, training=self.training)

        attention_values: MultiHeadedEmbeds = matmul(attention_probs, value)

        attention_values: SequenceInputEmbeds = self.reshape_to_head_insensitive(attention_values)

        attention_output: SequenceInputEmbeds = self.w_o(attention_values)

        return FCABSAttentionOutput(attention_output, position_bias, mask_drop)


class FCABSEncoder(BaselineEncoder):
    ATTENTION_CLS = FCABSAttention

    def forward(self, input_embeds: SequenceInputEmbeds, attention_mask: SequenceInputIds) -> AttentionOutput:
        config: TransformerConfig = self.config

        input_embeds = dropout(input_embeds, p=config.dropout_prob, training=self.training)

        position_bias = None
        for i in range(config.num_layers // 2):
            self_attention_layernorm: LayerNorm = self.layernorms[2 * i]
            self_attention_layer: AttentionBase = self.self_attention_layers[i]
            feedforward_layernorm: LayerNorm = self.layernorms[2 * i + 1]
            feedforward_layer: TransformerComponentBase = self.feedforward_layers[i]

            normed_input_embeds: SequenceInputEmbeds = self_attention_layernorm(input_embeds)

            # Also take the mask_drop here
            self_attention_output: FCABSAttentionOutput = self_attention_layer(normed_input_embeds, attention_mask, position_bias)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(self_attention_output.input_embeds, p=config.dropout_prob, training=self.training)
            position_bias = self_attention_output.position_bias

            # Update L dimension by dropping the indices which had the lowest attention activation, calculated earlier in self attention
            input_embeds, attention_mask, position_bias = update_L_dimension(input_embeds, attention_mask, position_bias, self_attention_output.mask_drop)
            

            normed_input_embeds: SequenceInputEmbeds = feedforward_layernorm(input_embeds)
            feedforward_output: SequenceInputEmbeds = feedforward_layer(normed_input_embeds)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(feedforward_output, p=config.dropout_prob, training=self.training)


        input_embeds: SequenceInputEmbeds = self.layernorms[-1](input_embeds)
        input_embeds = dropout(input_embeds, p=config.dropout_prob, training=self.training)

        return FCABSEncoderOutput(input_embeds=input_embeds, position_bias=None, attention_mask=attention_mask)



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

        input_embeds: SequenceInputEmbeds = embedding(input_ids, self.embedding.weight)
        encoder_outputs: FCABSEncoderOutput = self.encoder(input_embeds, attention_mask)
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

        return Seq2SeqLMOutput(loss=loss, logits=lm_logits)

