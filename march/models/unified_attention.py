from torch import concatenate, tile

from march.models.utils import *
from march.models.baseline import *


class UnifiedAttention(AttentionBase):
    def __init__(self, config: TransformerConfig, is_cross_attention: bool) -> None:
        super().__init__(config, is_cross_attention)

        self.w_q = Linear(config.dim_model, config.num_heads * config.dim_qkv, bias=False)
        self.w_k = Linear(config.dim_model, config.num_heads * config.dim_qkv, bias=False)
        self.w_v = Linear(config.dim_model, config.num_heads * config.dim_qkv, bias=False)
        self.w_o = Linear(config.num_heads * config.dim_qkv, config.dim_model, bias=False)

    def init_weights(self) -> None:
        config = self.config

        self.w_q.weight.data.normal_(mean=0.0, std=(config.dim_model * config.dim_qkv) ** -0.5)
        self.w_k.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)
        self.w_v.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)
        self.w_o.weight.data.normal_(mean=0.0, std=(config.num_heads * config.dim_qkv) ** -0.5)

    def forward(
        self,
        input_embeds: SequenceInputEmbeds,
        attention_mask: SequenceInputIds,
        encoder_key_value_states: KeyValueStates=None,
        encoder_attention_mask: SequenceInputIds=None,
    ) -> AttentionOutput:
        config = self.config

        ###############################
        # START Unified self and cross attention
        ###############################

        attention_values: List[SequenceInputEmbeds] = self.w_q(input_embeds), self.w_k(input_embeds), self.w_v(input_embeds)
        query, key, value = list(map(self.reshape_to_head_sensitive, attention_values))
        if self.is_cross_attention:
            cross_attention_key, cross_attention_value = encoder_key_value_states
            key: MultiHeadedEmbeds = concatenate((key, cross_attention_key), dim=2)
            value: MultiHeadedEmbeds = concatenate((value, cross_attention_value), dim=2)

        ###############################
        # END Unified self and cross attention
        ###############################

        attention_logits: MultiHeadedAttention = matmul(query, key.transpose(2, 3))

        ###############################
        # START Unified self and cross attention
        ###############################

        if self.is_cross_attention:
            batch_size, L_in = encoder_attention_mask.size()
            L_out = attention_mask.size()[1]

            self_attention_mask = encoder_attention_mask.reshape(batch_size, 1, 1, L_in)
            self_attention_mask = tile(self_attention_mask, (1, 1, L_out, 1))
            cross_attention_mask = attention_mask.reshape(batch_size, 1, L_out, L_out)

            attention_mask = concatenate((self_attention_mask, cross_attention_mask), dim=3)
        else:
            batch_size, sequence_length = attention_mask.size()
            attention_mask = attention_mask.reshape(batch_size, 1, 1, sequence_length)

        ###############################
        # END Unified self and cross attention
        ###############################

        attention_mask = attention_mask.to(attention_logits.dtype) * finfo(attention_logits.dtype).min
        attention_logits: MultiHeadedAttention = attention_logits + attention_mask

        attention_probs: MultiHeadedAttention = attention_logits.softmax(dim=3)
        attention_probs: MultiHeadedAttention = dropout(attention_probs, p=config.dropout_prob, training=self.training)

        attention_values: MultiHeadedEmbeds = matmul(attention_probs, value)

        attention_values: SequenceInputEmbeds = self.reshape_to_head_insensitive(attention_values)

        attention_output: SequenceInputEmbeds = self.w_o(attention_values)

        return AttentionOutput(attention_output, (key, value))


class UnifiedAttentionDecoderBase(DecoderBase):
    def __init__(self, config: TransformerConfig) -> None:
        TransformerComponentBase.__init__(self, config)

        self.layernorms = ModuleList([LayerNorm(config) for _ in range(config.num_layers + 1)])
        self.cross_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(config, is_cross_attention=True) for _ in range(config.num_layers // 2)]
        )
        self.feedforward_layers: List[TransformerComponentBase] = ModuleList(
            [self.FEEDFORWARD_CLS(config) for _ in range(config.num_layers // 2)]
        )

    def forward(
        self,
        input_embeds: SequenceInputEmbeds,
        attention_mask: SequenceInputIds,
        encoder_key_value_states: List[KeyValueStates],
        encoder_attention_mask: SequenceInputIds,
    ) -> AttentionOutput:
        config: TransformerConfig = self.config

        for i in range(config.num_layers // 2):
            cross_attention_layernorm: LayerNorm = self.layernorms[2 * i]
            cross_attention_key_value_states = encoder_key_value_states[i]
            cross_attention_layer: AttentionBase = self.cross_attention_layers[i]
            feedforward_layernorm: LayerNorm = self.layernorms[2 * i + 1]
            feedforward_layer: TransformerComponentBase = self.feedforward_layers[i]

            normed_input_embeds: SequenceInputEmbeds = cross_attention_layernorm(input_embeds)
            cross_attention_output: AttentionOutput = cross_attention_layer(normed_input_embeds, attention_mask, cross_attention_key_value_states, encoder_attention_mask)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(cross_attention_output.input_embeds, p=config.dropout_prob, training=self.training)

            normed_input_embeds: SequenceInputEmbeds = feedforward_layernorm(input_embeds)
            feedforward_output: SequenceInputEmbeds = feedforward_layer(normed_input_embeds)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(feedforward_output, p=config.dropout_prob, training=self.training)

        input_embeds: SequenceInputEmbeds = self.layernorms[-1](input_embeds)

        return AttentionOutput(input_embeds=input_embeds, key_value_states=None)


class UnifiedAttentionDecoder(UnifiedAttentionDecoderBase):
    ATTENTION_CLS = UnifiedAttention
    FEEDFORWARD_CLS = BaselineFeedforward


class UnifiedAttentionTransformer(TransformerBase):
    POSITION_ENCODING_CLS = AbsolutePositionEncoding
    ENCODER_CLS = BaselineEncoder
    DECODER_CLS = UnifiedAttentionDecoder
