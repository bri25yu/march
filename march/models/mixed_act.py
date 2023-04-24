from torch.nn.functional import gelu, relu, silu, sigmoid

from march.models.utils import *
from march.models.baseline import *
from march.models.absolute_position_embeddings import AbsolutePositionEncodingUnitVariance


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

        self.init_weights()

    def init_weights(self) -> None:
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
    POSITION_ENCODING_CLS = AbsolutePositionEncodingUnitVariance
    ENCODER_CLS = GatedLinearUnitEncoder
    DECODER_CLS = GatedLinearUnitDecoder


class MixedActFeedforward(TransformerComponentBase):
    def __init__(self, config: TransformerConfig) -> None:
        TransformerComponentBase.__init__(self, config)

        self.up_projection_slow = Linear(config.dim_model, config.dim_feedforward, bias=False)
        self.up_projection_fast = Linear(config.dim_model, config.dim_feedforward, bias=False)
        self.down_projection = Linear(config.dim_feedforward, config.dim_model, bias=False)

        self.init_weights()

    def init_weights(self) -> None:
        config = self.config

        self.up_projection_slow.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)
        self.up_projection_fast.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)
        self.down_projection.weight.data.normal_(mean=0.0, std=config.dim_feedforward ** -0.5)

    def forward(self, input_embeds: SequenceInputEmbeds) -> SequenceInputEmbeds:
        config = self.config

        input_embeds_slow: SequenceInputEmbeds = sigmoid(self.up_projection_slow(input_embeds))
        input_embeds_fast: SequenceInputEmbeds = relu(self.up_projection_fast(input_embeds))
        input_embeds: SequenceInputEmbeds = (input_embeds_slow + input_embeds_fast) / 2
        input_embeds: SequenceInputEmbeds = dropout(input_embeds, config.dropout_prob, training=self.training)
        input_embeds: SequenceInputEmbeds = self.down_projection(input_embeds)

        return input_embeds


class MixedActEncoder(EncoderBase):
    ATTENTION_CLS = BaselineAttention
    FEEDFORWARD_CLS = MixedActFeedforward


class MixedActDecoder(DecoderBase):
    ATTENTION_CLS = BaselineAttention
    FEEDFORWARD_CLS = MixedActFeedforward


class MixedActTransformer(TransformerBase):
    POSITION_ENCODING_CLS = AbsolutePositionEncodingUnitVariance
    ENCODER_CLS = MixedActEncoder
    DECODER_CLS = MixedActDecoder


class MixedActSumOverMeanFeedforward(TransformerComponentBase):
    def __init__(self, config: TransformerConfig) -> None:
        TransformerComponentBase.__init__(self, config)

        self.up_projection_slow = Linear(config.dim_model, config.dim_feedforward, bias=False)
        self.up_projection_fast = Linear(config.dim_model, config.dim_feedforward, bias=False)
        self.down_projection = Linear(config.dim_feedforward, config.dim_model, bias=False)

        self.init_weights()

    def init_weights(self) -> None:
        config = self.config

        self.up_projection_slow.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)
        self.up_projection_fast.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)
        self.down_projection.weight.data.normal_(mean=0.0, std=config.dim_feedforward ** -0.5)

    def forward(self, input_embeds: SequenceInputEmbeds) -> SequenceInputEmbeds:
        config = self.config

        input_embeds_slow: SequenceInputEmbeds = sigmoid(self.up_projection_slow(input_embeds))
        input_embeds_fast: SequenceInputEmbeds = relu(self.up_projection_fast(input_embeds))
        input_embeds: SequenceInputEmbeds = input_embeds_slow + input_embeds_fast
        input_embeds: SequenceInputEmbeds = dropout(input_embeds, config.dropout_prob, training=self.training)
        input_embeds: SequenceInputEmbeds = self.down_projection(input_embeds)

        return input_embeds


class MixedActSumOverMeanEncoder(EncoderBase):
    ATTENTION_CLS = BaselineAttention
    FEEDFORWARD_CLS = MixedActSumOverMeanFeedforward


class MixedActSumOverMeanDecoder(DecoderBase):
    ATTENTION_CLS = BaselineAttention
    FEEDFORWARD_CLS = MixedActSumOverMeanFeedforward


class MixedActSumOverMeanTransformer(TransformerBase):
    POSITION_ENCODING_CLS = AbsolutePositionEncodingUnitVariance
    ENCODER_CLS = MixedActSumOverMeanEncoder
    DECODER_CLS = MixedActSumOverMeanDecoder


class MixedActSOMDropoutFeedforward(MixedActSumOverMeanFeedforward):
    def forward(self, input_embeds: SequenceInputEmbeds) -> SequenceInputEmbeds:
        config = self.config

        input_embeds_slow: SequenceInputEmbeds = sigmoid(self.up_projection_slow(input_embeds))
        input_embeds_fast: SequenceInputEmbeds = relu(self.up_projection_fast(input_embeds))
        input_embeds: SequenceInputEmbeds = dropout(input_embeds_slow, 0.5, training=self.training) + input_embeds_fast
        input_embeds: SequenceInputEmbeds = dropout(input_embeds, config.dropout_prob, training=self.training)
        input_embeds: SequenceInputEmbeds = self.down_projection(input_embeds)

        return input_embeds


class MixedActSOMDropoutEncoder(EncoderBase):
    ATTENTION_CLS = BaselineAttention
    FEEDFORWARD_CLS = MixedActSOMDropoutFeedforward


class MixedActSOMDropoutDecoder(DecoderBase):
    ATTENTION_CLS = BaselineAttention
    FEEDFORWARD_CLS = MixedActSOMDropoutFeedforward


class MixedActSOMDropoutTransformer(TransformerBase):
    POSITION_ENCODING_CLS = AbsolutePositionEncodingUnitVariance
    ENCODER_CLS = MixedActSOMDropoutEncoder
    DECODER_CLS = MixedActSOMDropoutDecoder
