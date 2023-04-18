from march.models.utils import *
from march.models.baseline import *


class AbsolutePositionEncodingSumOverAverage(AbsolutePositionEncoding):
    def forward(self, inputs: SequenceInputEmbeds) -> SequenceInputEmbeds:
        inputs: SequenceInputEmbeds = dropout(inputs, p=self.config.dropout_prob, training=self.training)

        sequence_length = inputs.size()[1]
        timing: SequenceInputEmbeds = self.timing_table[None, :sequence_length, :]
        timing: SequenceInputEmbeds = dropout(timing, p=self.config.dropout_prob, training=self.training)

        return inputs + timing


class AbsolutePositionEncodingUnitVariance(AbsolutePositionEncoding):
    def init_weights(self) -> None:
        self.timing_table.data.normal_(mean=0.0, std=1.0)


class APESumOverAverageTransformer(TransformerBase):
    POSITION_ENCODING_CLS = AbsolutePositionEncodingSumOverAverage
    ENCODER_CLS = BaselineEncoder
    DECODER_CLS = BaselineDecoder


class APEUnitVarianceTransformer(TransformerBase):
    POSITION_ENCODING_CLS = AbsolutePositionEncodingUnitVariance
    ENCODER_CLS = BaselineEncoder
    DECODER_CLS = BaselineDecoder
