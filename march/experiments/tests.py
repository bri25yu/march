from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig
from march.models.scaling_heads import ScalingHeadsTransformer, InverseScalingHeadsTransformer
from march.models.unified_attention import UnifiedAttentionTransformer
from march.models.scaling_heads_constant import ScalingHeadsConstantTransformer, InverseScalingHeadsConstantTransformer
from march.models.database import DatabaseTransformerConfig, DatabaseTransformer
from march.models.absolute_position_embeddings import APESumOverAverageTransformer, APEUnitVarianceTransformer

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
        config = DatabaseTransformerConfig()
        config.num_database_states = config.dim_model * (config.num_layers // 2)

        config.num_heads = config.num_heads // 2
        return DatabaseTransformer(config)


class DatabaseFromLayersExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = DatabaseTransformerConfig(num_layers=4)
        config.num_database_states = config.dim_model * (config.num_layers // 2)

        return DatabaseTransformer(config)


class DatabaseFromDimExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = DatabaseTransformerConfig(dim_model=448)
        config.num_database_states = config.dim_model * (config.num_layers // 2)

        return DatabaseTransformer(config)


class DatabaseExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = DatabaseTransformerConfig()
        config.num_database_states = config.dim_model
        return DatabaseTransformer(config)


class DatabaseFromHeads2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = DatabaseTransformerConfig()
        config.num_database_states = config.dim_model

        config.num_heads = config.num_heads - 1
        return DatabaseTransformer(config)


class APESumOverAverageExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return APESumOverAverageTransformer(config)


class APEUnitVarianceExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return APEUnitVarianceTransformer(config)


class MoreHeadsLessQKVDimLessLayersExperiment(APEUnitVarianceExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_qkv=32, num_layers=4, num_heads=24)
        return APEUnitVarianceTransformer(config)
