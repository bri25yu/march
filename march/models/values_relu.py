from typing import List, Type

from march.models.baseline import *
from march.models.utils import *
from march.models.no_ff import NoFFDecoder, NoFFEncoder
from torch.nn.functional import dropout, embedding, relu

"""
Add the relu back into the no_ff model but in the values transformation
part of the attention mechanism. So there will be a nonlinearity again 
but it will be transforming the inputs before they are linearly recombined
in the attention mechanism.
"""

__all__ = ["ValuesReluTransformer", "ValuesReluFirstFFTransformer"]


# This is a copy of the BaselineFeedforward class
class ValuesFF(TransformerComponentBase):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)

        self.up_projection = Linear(
            config.dim_model, config.dim_feedforward, bias=False
        )
        self.down_projection = Linear(
            config.dim_feedforward, config.dim_model, bias=False
        )

    def _init_weights(self) -> None:
        # We use _init_weights rather than init_weights to have the encoder/decoder handle all the weight inits
        config = self.config

        self.up_projection.weight.data.normal_(mean=0.0, std=config.dim_model**-0.5)
        self.down_projection.weight.data.normal_(
            mean=0.0, std=config.dim_feedforward**-0.5
        )

    def forward(self, input_embeds: SequenceInputEmbeds) -> SequenceInputEmbeds:
        config = self.config

        input_embeds: SequenceInputEmbeds = self.up_projection(input_embeds)
        input_embeds: SequenceInputEmbeds = relu(input_embeds)
        input_embeds: SequenceInputEmbeds = dropout(
            input_embeds, config.dropout_prob, training=self.training
        )
        input_embeds: SequenceInputEmbeds = self.down_projection(input_embeds)

        return input_embeds


# Only one projection after the relu
class ValuesReluFirstFF(TransformerComponentBase):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)

        self.projection = Linear(config.dim_model, config.dim_model, bias=False)

    def _init_weights(self) -> None:
        # We use _init_weights rather than init_weights to have the encoder/decoder handle all the weight inits
        config = self.config

        self.projection.weight.data.normal_(mean=0.0, std=config.dim_model**-0.5)

    def forward(self, input_embeds: SequenceInputEmbeds) -> SequenceInputEmbeds:
        config = self.config

        input_embeds: SequenceInputEmbeds = dropout(
            input_embeds, config.dropout_prob, training=self.training
        )
        input_embeds: SequenceInputEmbeds = relu(input_embeds)
        input_embeds: SequenceInputEmbeds = self.projection(input_embeds)

        return input_embeds


class ValuesReluAttention(BaselineAttention):
    def __init__(
        self,
        config: TransformerConfig,
        is_cross_attention: bool,
        has_relative_attention_bias: bool = False,
    ) -> None:
        super().__init__(config, is_cross_attention, has_relative_attention_bias)
        # If self attention, init extra values transform:
        if not is_cross_attention:
            self.values_ff = ValuesFF(config)

    def _init_weights(self) -> None:
        super()._init_weights()
        if not self.is_cross_attention:
            self.values_ff._init_weights()

    # TODO: update with new baseline class
    def forward(
        self,
        input_embeds: SequenceInputEmbeds,
        attention_mask: SequenceInputIds = None,
        position_bias: MultiHeadedAttention = None,
        encoder_hidden_state: SequenceInputEmbeds = None,
    ) -> AttentionOutput:
        config = self.config

        if not self.is_cross_attention:
            attention_values: List[SequenceInputEmbeds] = (
                self.w_q(input_embeds),
                self.w_k(input_embeds),
                self.w_v(input_embeds),
            )
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
            attention_mask = (
                attention_mask.to(attention_logits.dtype)
                * finfo(attention_logits.dtype).min
            )
            attention_logits: MultiHeadedAttention = attention_logits + attention_mask

        if position_bias is not None:
            attention_logits: MultiHeadedAttention = attention_logits + position_bias
        elif self.has_relative_attention_bias:
            position_bias = self.compute_bias(query_length, key_length, is_decoder)
            attention_logits: MultiHeadedAttention = attention_logits + position_bias

        attention_probs: MultiHeadedAttention = attention_logits.softmax(dim=3)
        attention_probs: MultiHeadedAttention = dropout(
            attention_probs, p=config.dropout_prob, training=self.training
        )

        # Apply nonlinearity to values
        if not self.is_cross_attention:
            # Doing this again will make it slower, but can fix later if it does well:
            value = self.reshape_to_head_insensitive(value)
            # apply ff and relu
            value = self.values_ff(value)
            value = self.reshape_to_head_sensitive(value)

        attention_values: MultiHeadedEmbeds = matmul(attention_probs, value)

        attention_values: SequenceInputEmbeds = self.reshape_to_head_insensitive(
            attention_values
        )

        attention_output: SequenceInputEmbeds = self.w_o(attention_values)

        return AttentionOutput(attention_output, position_bias)


class ValuesReluFirstFFAttention(ValuesReluAttention):
    def __init__(
        self,
        config: TransformerConfig,
        is_cross_attention: bool,
        has_relative_attention_bias: bool = False,
    ) -> None:
        super().__init__(config, is_cross_attention, has_relative_attention_bias)
        # If self attention, init extra values transform:
        if not is_cross_attention:
            self.values_ff = ValuesReluFirstFF(config)


class ValuesReluEncoder(NoFFEncoder):
    ATTENTION_CLS = ValuesReluAttention


class ValuesReluDecoder(NoFFDecoder):
    ATTENTION_CLS = ValuesReluAttention


class ValuesReluTransformer(BaselineTransformer):
    ENCODER_CLS = ValuesReluEncoder
    DECODER_CLS = ValuesReluDecoder


class ValuesReluFirstFFEncoder(NoFFEncoder):
    ATTENTION_CLS = ValuesReluFirstFFAttention


class ValuesReluFirstFFDecoder(NoFFDecoder):
    ATTENTION_CLS = ValuesReluFirstFFAttention


class ValuesReluFirstFFTransformer(BaselineTransformer):
    ENCODER_CLS = ValuesReluFirstFFEncoder
    DECODER_CLS = ValuesReluFirstFFDecoder
