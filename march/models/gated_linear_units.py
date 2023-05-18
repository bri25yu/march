from torch.nn.functional import gelu, relu, silu

from march.models.utils import *
from march.models.baseline import *


class GateFunctions:
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"


STR_TO_GATE_FN = {
    GateFunctions.RELU: relu,
    GateFunctions.GELU: gelu,
    GateFunctions.SILU: silu,
}


@dataclass
class GatedLinearUnitTransformerConfig(TransformerConfig):
    gate_fn: Union[None, str] = None


class GatedLinearUnitFeedforward(TransformerComponentBase):
    def __init__(self, config: GatedLinearUnitTransformerConfig) -> None:
        super().__init__(config)

        self.up_projection = Linear(config.dim_model, config.dim_feedforward, bias=False)
        self.gate_projection = Linear(config.dim_model, config.dim_feedforward, bias=False)
        self.down_projection = Linear(config.dim_feedforward, config.dim_model, bias=False)

    def _init_weights(self) -> None:
        config = self.config

        self.up_projection.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)
        self.gate_projection.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)
        self.down_projection.weight.data.normal_(mean=0.0, std=config.dim_feedforward ** -0.5)

    def forward(self, input_embeds: SequenceInputEmbeds) -> SequenceInputEmbeds:
        config: GatedLinearUnitTransformerConfig = self.config
        gate_fn = STR_TO_GATE_FN[config.gate_fn]

        gate: SequenceInputEmbeds = gate_fn(self.gate_projection(input_embeds))
        input_embeds: SequenceInputEmbeds = self.up_projection(input_embeds)
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
    ENCODER_CLS = GatedLinearUnitEncoder
    DECODER_CLS = GatedLinearUnitDecoder
