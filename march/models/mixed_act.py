from typing import Any

from torch.nn.functional import gelu, relu, silu

from march.models.utils import *
from march.models.baseline import *
from march.models.absolute_position_embeddings import AbsolutePositionEncodingUnitVariance


class GateFunctions:
    RELU = relu
    GELU = gelu
    SILU = silu


@dataclass
class GatedLinearUnitTransformerConfig(TransformerConfig):
    gate_fn: Any = None  # TODO Type check that gate_fn is a member of `GateFunctions`


class GatedLinearUnitFeedforward(BaselineFeedforward):
    def __init__(self, config: GatedLinearUnitTransformerConfig) -> None:
        TransformerComponentBase.__init__(self, config)

        self.up_projection = Linear(config.dim_model, config.dim_feedforward, bias=False, dtype=MODEL_PRECISION)
        self.gate_projection = Linear(config.dim_model, config.dim_feedforward, bias=False, dtype=MODEL_PRECISION)
        self.down_projection = Linear(config.dim_feedforward, config.dim_model, bias=False, dtype=MODEL_PRECISION)

    def init_weights(self) -> None:
        config = self.config

        self.up_projection.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)
        self.gate_projection.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)
        self.down_projection.weight.data.normal_(mean=0.0, std=config.dim_feedforward ** -0.5)

    def forward(self, input_embeds: SequenceInputEmbeds) -> SequenceInputEmbeds:
        config: GatedLinearUnitTransformerConfig = self.config

        input_embeds: SequenceInputEmbeds = self.up_projection(input_embeds)
        gate: SequenceInputEmbeds = config.gate_fn(input_embeds)
        input_embeds: SequenceInputEmbeds = input_embeds * gate
        input_embeds: SequenceInputEmbeds = dropout(input_embeds, config.dropout_prob, training=self.training)
        input_embeds: SequenceInputEmbeds = self.down_projection(input_embeds)

        return input_embeds


class GatedLinearUnitEncoder(EncoderBase):
    ATTENTION_CLS = BaselineAttention
    FEEDFORWARD_CLS = GatedLinearUnitFeedforward


class GatedLinearUnitDecoder(DecoderBase):
    ATTENTION_CLS = BaselineAttention
    FEEDFORWARD_CLS = GatedLinearUnitFeedforward


class GatedLinearUnitTransformer(TransformerBase):
    POSITION_ENCODING_CLS = AbsolutePositionEncodingUnitVariance
    ENCODER_CLS = GatedLinearUnitEncoder
    DECODER_CLS = GatedLinearUnitDecoder
