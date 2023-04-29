from march.models.scaling_heads import ScalingHeadsTransformer, InverseScalingHeadsTransformer
from march.models.scaling_heads_constant import ScalingHeadsConstantTransformer, InverseScalingHeadsConstantTransformer

from march.experiments.baseline import *


class ScalingHeadsExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return ScalingHeadsTransformer(config)


class InverseScalingHeadsExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return InverseScalingHeadsTransformer(config)


class ScalingHeadsConstantExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return ScalingHeadsConstantTransformer(config)


class InverseScalingHeadsConstantExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return InverseScalingHeadsConstantTransformer(config)
