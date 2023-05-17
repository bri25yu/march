from typing import List, Type

from march.models.baseline import *
from march.models.utils import *


__all__ = ["NoFFTransformer"]


class NoFFEncoder(TransformerComponentBase):
    ATTENTION_CLS = BaselineAttention

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)

        # Only half the amount of layernorms (still + 1 for first one)
        self.layernorms = ModuleList([LayerNorm(config) for _ in range(config.num_layers // 2 + 1)])
        self.self_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(config, is_cross_attention=False, has_relative_attention_bias=i==0) for i in range(config.num_layers // 2)]
        )
        # self.feedforward_layers: List[TransformerComponentBase] = ModuleList(
        #     [self.FEEDFORWARD_CLS(config) for _ in range(config.num_layers // 2)]
        # )

    def forward(self, input_embeds: SequenceInputEmbeds, attention_mask: SequenceInputIds) -> AttentionOutput:
        config: TransformerConfig = self.config

        position_bias = None
        for i in range(config.num_layers // 2):
            self_attention_layernorm: LayerNorm = self.layernorms[i]
            self_attention_layer: AttentionBase = self.self_attention_layers[i]
            # Get rid of ff layer
            # feedforward_layernorm: LayerNorm = self.layernorms[2 * i + 1]
            # feedforward_layer: TransformerComponentBase = self.feedforward_layers[i]

            normed_input_embeds: SequenceInputEmbeds = self_attention_layernorm(input_embeds)
            self_attention_output: AttentionOutput = self_attention_layer(normed_input_embeds, attention_mask, position_bias)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(self_attention_output.input_embeds, p=config.dropout_prob, training=self.training)
            position_bias = self_attention_output.position_bias

            # Get rid of ff layer
            # normed_input_embeds: SequenceInputEmbeds = feedforward_layernorm(input_embeds)
            # feedforward_output: SequenceInputEmbeds = feedforward_layer(normed_input_embeds)
            # input_embeds: SequenceInputEmbeds = input_embeds + dropout(feedforward_output, p=config.dropout_prob, training=self.training)

        input_embeds: SequenceInputEmbeds = self.layernorms[-1](input_embeds)

        return AttentionOutput(input_embeds=input_embeds, position_bias=None)



class NoFFDecoder(TransformerComponentBase):
    ATTENTION_CLS = BaselineAttention

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)

        # Only half the amount of layernorms (not / 2 * 3) (still + 1 for first one)
        self.layernorms = ModuleList([LayerNorm(config) for _ in range(config.num_layers + 1)])
        self.self_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(config, is_cross_attention=False, has_relative_attention_bias=i==0) for i in range(config.num_layers // 2)]
        )
        self.cross_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(config, is_cross_attention=True) for _ in range(config.num_layers // 2)]
        )
        # self.feedforward_layers: List[TransformerComponentBase] = ModuleList(
        #     [self.FEEDFORWARD_CLS(config) for _ in range(config.num_layers // 2)]
        # )

    def forward(
        self,
        input_embeds: SequenceInputEmbeds,
        attention_mask: SequenceInputIds,
        encoder_hidden_state: SequenceInputEmbeds,
        encoder_attention_mask: SequenceInputIds,
    ) -> AttentionOutput:
        config: TransformerConfig = self.config

        position_bias = None
        for i in range(config.num_layers // 2):
            self_attention_layernorm: LayerNorm = self.layernorms[2 * i]
            self_attention_layer: AttentionBase = self.self_attention_layers[i]
            cross_attention_layernorm: LayerNorm = self.layernorms[2 * i + 1]
            cross_attention_layer: AttentionBase = self.cross_attention_layers[i]
            # Get rid of ff layer
            # feedforward_layernorm: LayerNorm = self.layernorms[2 * i + 2]
            # feedforward_layer: TransformerComponentBase = self.feedforward_layers[i]

            normed_input_embeds: SequenceInputEmbeds = self_attention_layernorm(input_embeds)
            self_attention_output: AttentionOutput = self_attention_layer(normed_input_embeds, attention_mask, position_bias)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(self_attention_output.input_embeds, p=config.dropout_prob, training=self.training)
            position_bias = self_attention_output.position_bias

            normed_input_embeds: SequenceInputEmbeds = cross_attention_layernorm(input_embeds)
            cross_attention_output: AttentionOutput = cross_attention_layer(normed_input_embeds, encoder_attention_mask, encoder_hidden_state=encoder_hidden_state)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(cross_attention_output.input_embeds, p=config.dropout_prob, training=self.training)

            # Get rid of ff layer
            # normed_input_embeds: SequenceInputEmbeds = feedforward_layernorm(input_embeds)
            # feedforward_output: SequenceInputEmbeds = feedforward_layer(normed_input_embeds)
            # input_embeds: SequenceInputEmbeds = input_embeds + dropout(feedforward_output, p=config.dropout_prob, training=self.training)

        input_embeds: SequenceInputEmbeds = self.layernorms[-1](input_embeds)

        return AttentionOutput(input_embeds=input_embeds, position_bias=None)


class NoFFTransformer(TransformerBase):
    ENCODER_CLS = NoFFEncoder
    DECODER_CLS = NoFFDecoder

