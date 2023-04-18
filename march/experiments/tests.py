from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig
from march.models.scaling_heads import ScalingHeadsTransformer, InverseScalingHeadsTransformer
from march.models.unified_attention import UnifiedAttentionTransformer
from march.models.scaling_heads_constant import ScalingHeadsConstantTransformer, InverseScalingHeadsConstantTransformer
from march.models.database import DatabaseTransformer

from march.experiments.baseline import BaselineExperiment


class MoreHeadsLessLayersExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(num_layers=4, num_heads=16)
        return BaselineTransformer(config)


class MoreHeadsLessQKVDimExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_qkv=32)
        return BaselineTransformer(config)


class LessHeadsMoreQKVDimExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_qkv=128)
        return BaselineTransformer(config)


class ScalingHeadsExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return ScalingHeadsTransformer(config)


class InverseScalingHeadsExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return InverseScalingHeadsTransformer(config)


class UnifiedAttentionExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(num_heads=12)
        return UnifiedAttentionTransformer(config)


class ScalingHeadsConstantExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return ScalingHeadsConstantTransformer(config)


class InverseScalingHeadsConstantExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return InverseScalingHeadsConstantTransformer(config)


class DatabaseFromHeadsExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_heads = config.num_heads // 2
        return DatabaseTransformer(config)


class DatabaseFromLayersExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(num_layers=4)
        return DatabaseTransformer(config)
