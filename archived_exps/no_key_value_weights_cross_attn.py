from march.models.baseline import *
from march.models.utils import *


__all__ = ["NoKeyValueWeightsCrossAttentionTransformer"]


# TODO update with new attn relative position bias
class NoKeyValueWeightsCrossAttention(AttentionBase):
    def __init__(self, config: TransformerConfig, is_cross_attention: bool) -> None:
        super().__init__(config, is_cross_attention)

        self.w_q = Linear(config.dim_model, config.num_heads * config.dim_qkv, bias=False)
        # No w_k or w_v
        # self.w_k = Linear(config.dim_model, config.num_heads * config.dim_qkv, bias=False)
        # self.w_v = Linear(config.dim_model, config.num_heads * config.dim_qkv, bias=False)
        self.w_o = Linear(config.num_heads * config.dim_qkv, config.dim_model, bias=False)

    def init_weights(self) -> None:
        config = self.config

        self.w_q.weight.data.normal_(mean=0.0, std=(config.dim_model * config.dim_qkv) ** -0.5)
        # No w_k or w_v
        # self.w_k.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)
        # self.w_v.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)
        self.w_o.weight.data.normal_(mean=0.0, std=(config.num_heads * config.dim_qkv) ** -0.5)

    def forward(
        self,
        input_embeds: SequenceInputEmbeds,
        attention_mask: SequenceInputIds=None,
        encoder_key_value_states: KeyValueStates=None,
    ) -> AttentionOutput:
        config = self.config

        if not self.is_cross_attention:
            attention_values: List[SequenceInputEmbeds] = self.w_q(input_embeds), self.w_k(input_embeds), self.w_v(input_embeds)
            # This call moved here to remove some reshaping operations later in the cross attn case.
            query, key, value = list(map(self.reshape_to_head_sensitive, attention_values))

        else:
            key, value = encoder_key_value_states

            query: SequenceInputEmbeds = self.w_q(input_embeds)

            # Change is here: Removing the key and value weight applications
            # key: SequenceInputEmbeds = self.w_k(self.reshape_to_head_insensitive(key))
            # value: SequenceInputEmbeds = self.w_v(self.reshape_to_head_insensitive(value))

            attention_values: List[SequenceInputEmbeds] = (query, key, value)

            query = self.reshape_to_head_sensitive(query)

        # query, key, value = list(map(self.reshape_to_head_sensitive, attention_values))
        # The rest is the same

        attention_logits: MultiHeadedAttention = matmul(query, key.transpose(2, 3))

        if attention_mask is not None:
            if len(attention_mask.size()) == 2:
                query_length = 1
                batch_size, key_length = attention_mask.size()
            elif len(attention_mask.size()) == 3:
                batch_size, query_length, key_length = attention_mask.size()

            attention_mask = attention_mask.reshape(batch_size, 1, query_length, key_length)
            attention_mask = attention_mask.to(attention_logits.dtype) * finfo(attention_logits.dtype).min
            attention_logits: MultiHeadedAttention = attention_logits + attention_mask

        attention_probs: MultiHeadedAttention = attention_logits.softmax(dim=3)
        attention_probs: MultiHeadedAttention = dropout(attention_probs, p=config.dropout_prob, training=self.training)

        attention_values: MultiHeadedEmbeds = matmul(attention_probs, value)

        attention_values: SequenceInputEmbeds = self.reshape_to_head_insensitive(attention_values)

        attention_output: SequenceInputEmbeds = self.w_o(attention_values)

        return AttentionOutput(attention_output, (key, value))


class NoKeyValueWeightsCrossAttentionDecoder(DecoderBase):
    ATTENTION_CLS = BaselineAttention
    FEEDFORWARD_CLS = BaselineFeedforward

    # Use unique class here for cross attention, which doesn't have w_k or w_v weights.
    CROSS_ATTN_CLS: TransformerComponentBase = NoKeyValueWeightsCrossAttention

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)

        self.layernorms = ModuleList([LayerNorm(config) for _ in range(((config.num_layers // 2) * 3) + 1)])
        self.self_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(config, is_cross_attention=False) for _ in range(config.num_layers // 2)]
        )
        self.cross_attention_layers: List[AttentionBase] = ModuleList(
            [self.CROSS_ATTN_CLS(config, is_cross_attention=True) for _ in range(config.num_layers // 2)]
        )
        self.feedforward_layers: List[TransformerComponentBase] = ModuleList(
            [self.FEEDFORWARD_CLS(config) for _ in range(config.num_layers // 2)]
        )


class NoKeyValueWeightsCrossAttentionTransformer(TransformerBase):
    ENCODER_CLS = BaselineEncoder
    DECODER_CLS = NoKeyValueWeightsCrossAttentionDecoder
