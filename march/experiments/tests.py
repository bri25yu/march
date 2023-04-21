from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig
from march.models.scaling_heads import ScalingHeadsTransformer, InverseScalingHeadsTransformer
from march.models.unified_attention import UnifiedAttentionTransformer
from march.models.scaling_heads_constant import ScalingHeadsConstantTransformer, InverseScalingHeadsConstantTransformer
from march.models.database import DatabaseTransformerConfig, DatabaseTransformer
from march.models.absolute_position_embeddings import APESumOverAverageTransformer, APEUnitVarianceTransformer
from march.models.big_heads import BigHeadsTransformer, BigHeadsTransformerConfig
from march.models.mixed_act import GatedLinearUnitTransformer, GatedLinearUnitTransformerConfig, GateFunctions

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


class MoreHeadsLessQKVDimLessLayersExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_qkv=32, num_layers=4, num_heads=24)
        return APEUnitVarianceTransformer(config)


# Targeted experiments for BigHeadTransformer:
# 2x d_kv for each head but smaller overall hidden dimension 
# size due to constraint of keeping the model size the same
# dim model: 512 -> 448
# Num params = 35,582,400
# Num params baseline model = 36,340,224
class BigHeadsExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsTransformerConfig(dim_model=448,dim_qkv=112)
        return BigHeadsTransformer(config)


class ReLUGatedLinearUnitExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        dim_model = 512
        dim_feedforward = ((dim_model * 4) * 2) // 3
        config = GatedLinearUnitTransformerConfig(
            dim_model=dim_model, dim_feedforward=dim_feedforward, gate_fn=GateFunctions.RELU
        )
        return GatedLinearUnitTransformer(config)


class GELUGatedLinearUnitExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        dim_model = 512
        dim_feedforward = ((dim_model * 4) * 2) // 3
        config = GatedLinearUnitTransformerConfig(
            dim_model=dim_model, dim_feedforward=dim_feedforward, gate_fn=GateFunctions.GELU
        )
        return GatedLinearUnitTransformer(config)


class SiLUGatedLinearUnitExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        dim_model = 512
        dim_feedforward = ((dim_model * 4) * 2) // 3
        config = GatedLinearUnitTransformerConfig(
            dim_model=dim_model, dim_feedforward=dim_feedforward, gate_fn=GateFunctions.SILU
        )
        return GatedLinearUnitTransformer(config)
