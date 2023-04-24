from march.models.utils import *
from march.models.baseline import *


class NoSelfAttentionResidualEncoderBase(EncoderBase):
    # This is an exact copy of `EncoderBase.forward` unless specified otherwise
    def forward(self, input_embeds: SequenceInputEmbeds, attention_mask: SequenceInputIds) -> AttentionOutput:
        config: TransformerConfig = self.config

        encoder_key_value_states: List[KeyValueStates] = []
        for i in range(config.num_layers // 2):
            self_attention_layernorm: LayerNorm = self.layernorms[2 * i]
            self_attention_layer: AttentionBase = self.self_attention_layers[i]
            feedforward_layernorm: LayerNorm = self.layernorms[2 * i + 1]
            feedforward_layer: TransformerComponentBase = self.feedforward_layers[i]

            normed_input_embeds: SequenceInputEmbeds = self_attention_layernorm(input_embeds)
            self_attention_output: AttentionOutput = self_attention_layer(normed_input_embeds, attention_mask)

            ###########################
            # START No self attention residual
            ###########################

            # Original code:
            # input_embeds: SequenceInputEmbeds = input_embeds + dropout(self_attention_output.input_embeds, p=config.dropout_prob, training=self.training)

            input_embeds: SequenceInputEmbeds = dropout(self_attention_output.input_embeds, p=config.dropout_prob, training=self.training)

            ###########################
            # END No self attention residual
            ###########################

            encoder_key_value_states.append(self_attention_output.key_value_states)

            normed_input_embeds: SequenceInputEmbeds = feedforward_layernorm(input_embeds)
            feedforward_output: SequenceInputEmbeds = feedforward_layer(normed_input_embeds)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(feedforward_output, p=config.dropout_prob, training=self.training)

        input_embeds: SequenceInputEmbeds = self.layernorms[-1](input_embeds)

        return AttentionOutput(input_embeds=input_embeds, key_value_states=encoder_key_value_states)


class NoSelfAttentionResidualDecoderBase(DecoderBase):
    # This is an exact copy of `DecoderBase.forward` unless specified otherwise
    def forward(
        self,
        input_embeds: SequenceInputEmbeds,
        attention_mask: SequenceInputIds,
        encoder_key_value_states: List[KeyValueStates],
        encoder_attention_mask: SequenceInputIds,
    ) -> AttentionOutput:
        config: TransformerConfig = self.config

        for i in range(config.num_layers // 2):
            self_attention_layernorm: LayerNorm = self.layernorms[2 * i]
            self_attention_layer: AttentionBase = self.self_attention_layers[i]
            cross_attention_layernorm: LayerNorm = self.layernorms[2 * i + 1]
            cross_attention_key_value_states = encoder_key_value_states[i]
            cross_attention_layer: AttentionBase = self.cross_attention_layers[i]
            feedforward_layernorm: LayerNorm = self.layernorms[2 * i + 2]
            feedforward_layer: TransformerComponentBase = self.feedforward_layers[i]

            normed_input_embeds: SequenceInputEmbeds = self_attention_layernorm(input_embeds)
            self_attention_output: AttentionOutput = self_attention_layer(normed_input_embeds, attention_mask)

            ###########################
            # START No self attention residual
            ###########################

            # Original code:
            # input_embeds: SequenceInputEmbeds = input_embeds + dropout(self_attention_output.input_embeds, p=config.dropout_prob, training=self.training)

            input_embeds: SequenceInputEmbeds = dropout(self_attention_output.input_embeds, p=config.dropout_prob, training=self.training)

            ###########################
            # END No self attention residual
            ###########################

            normed_input_embeds: SequenceInputEmbeds = cross_attention_layernorm(input_embeds)
            cross_attention_output: AttentionOutput = cross_attention_layer(normed_input_embeds, encoder_attention_mask, cross_attention_key_value_states)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(cross_attention_output.input_embeds, p=config.dropout_prob, training=self.training)

            normed_input_embeds: SequenceInputEmbeds = feedforward_layernorm(input_embeds)
            feedforward_output: SequenceInputEmbeds = feedforward_layer(normed_input_embeds)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(feedforward_output, p=config.dropout_prob, training=self.training)

        input_embeds: SequenceInputEmbeds = self.layernorms[-1](input_embeds)

        return AttentionOutput(input_embeds=input_embeds, key_value_states=None)


class NoSelfAttentionResidualEncoder(NoSelfAttentionResidualEncoderBase):
    ATTENTION_CLS = BaselineAttention
    FEEDFORWARD_CLS = BaselineFeedforward


class NoSelfAttentionResidualDecoder(NoSelfAttentionResidualDecoderBase):
    ATTENTION_CLS = BaselineAttention
    FEEDFORWARD_CLS = BaselineFeedforward


class NoSelfAttentionResidualTransformer(TransformerBase):
    POSITION_ENCODING_CLS = AbsolutePositionEncoding
    ENCODER_CLS = NoSelfAttentionResidualEncoder
    DECODER_CLS = NoSelfAttentionResidualDecoder
