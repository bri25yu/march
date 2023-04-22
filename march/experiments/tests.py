from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig
from march.models.scaling_heads import ScalingHeadsTransformer, InverseScalingHeadsTransformer
from march.models.unified_attention import UnifiedAttentionTransformer
from march.models.scaling_heads_constant import ScalingHeadsConstantTransformer, InverseScalingHeadsConstantTransformer
from march.models.database import DatabaseTransformerConfig, DatabaseTransformer
from march.models.absolute_position_embeddings import APESumOverAverageTransformer, APEUnitVarianceTransformer
from march.models.big_heads import BigHeadsTransformer, BigHeadsTransformerConfig
from march.models.mixed_act import (
    GatedLinearUnitTransformer,
    GatedLinearUnitTransformerConfig,
    GateFunctions,
    MixedActTransformer,
    MixedActSumOverMeanTransformer,
    MixedActSOMDropoutTransformer,
)
from march.models.sparse_seqlen_attention import NoSelfAttentionResidualTransformer

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


# Targeted experiments for BigHeadsTransformer:
# 2x d_kv for each head but smaller overall hidden dimension 
# size due to constraint of keeping the model size the same
# dim model: 512 -> 448
# Num params = 35,582,400
# Num params baseline model = 36,340,224
# Head size D_kv = D_kv_orig x 2 = D_model / 4, scaling down D_model to 380
# w_o is from D_kv_orig x 2 x 8 -> D_model
# AKA w_o is from D_model x 2 -> D_model
# FF layer D_model -> 4 x D_model -> D_model
class BigHeadsExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsTransformerConfig(dim_model=448,dim_qkv=112,head_scale_size=2)
        return BaselineTransformer(config)


# Head size D_kv = D_kv_orig x 4 = D_model / 2, scaling down D_model to 380
# w_o is from D_kv_orig x 4 x 8 -> D_model
# AKA w_o is from D_model x 4 -> D_model
# FF layer D_model -> 4 x D_model -> D_model
class BigHeads2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsTransformerConfig(dim_model=380,dim_qkv=190,head_scale_size=4)
        return BaselineTransformer(config)


# Head size D_kv = D_kv_orig x 8 = D_model, scaling down D_model to 380
# w_o is from D_kv_orig x 8 x 8 -> D_model
# AKA w_o is from D_model x 8 -> D_model
# FF layer D_model -> 4 x D_model -> D_model
class BigHeads3Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsTransformerConfig(dim_model=380,dim_qkv=380,head_scale_size=8)
        return BaselineTransformer(config)


# Head size D_kv = D_kv_orig x 2 = D_model / 4, scaling down D_model to 448
# w_o is now from D_kv_orig x 2 x 8 -> D_kv_orig x 2 x 8
# AKA w_o is from D_model x 2 -> D_model x 2
# FF layer D_model x 2 -> 4 x D_model -> D_model
class BigHeadsLinearW_oExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsTransformerConfig(dim_model=448,dim_qkv=112,head_scale_size=2)
        return BigHeadsTransformer(config)


# Head size D_kv = D_kv_orig x 4 = D_model / 2, scaling down D_model to 448
# w_o is now from D_kv_orig x 4 x 8 -> D_kv_orig x 4 x 8
# AKA w_o is from D_model x 4 -> D_model x 4
# FF layer 4 x D_model -> 4 x D_model -> D_model
class BigHeadsLinearW_o2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsTransformerConfig(dim_model=380,dim_qkv=190,head_scale_size=4)
        return BigHeadsTransformer(config)


# Head size D_kv = D_kv_orig x 8 = D_model, scaling down D_model to 448
# w_o is now from D_kv_orig x 8 x 8-> D_kv_orig x 8 x 8
# AKA w_o is from D_model x 8 -> D_model x 8
# FF layer 8 x D_model -> 4 x D_model -> D_model
class BigHeadsLinearW_o3Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsTransformerConfig(dim_model=380,dim_qkv=380,head_scale_size=8)
        return BigHeadsTransformer(config)


# Head size D_kv = D_kv_orig x 4 = D_model / 2, scaling down D_model to 448
# w_o is now from D_kv_orig x 4 x 8 -> D_kv_orig x 4 x 8
# AKA w_o is from D_model x 4 -> D_model x 4
# FF layer 4 x D_model -> 2 x D_model -> D_model
class BigHeadsDownProjectExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsTransformerConfig(dim_model=380,dim_qkv=190,head_scale_size=4,feedforward_scale=2)
        return BigHeadsTransformer(config)


# Head size D_kv = D_kv_orig x 8 = D_model, scaling down D_model to 448
# w_o is now from D_kv_orig x 8 x 8-> D_kv_orig x 8 x 8
# AKA w_o is from D_model x 8 -> D_model x 4
# FF layer 4 x D_model -> 2 x D_model -> D_model
class BigHeadsDownProject2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsTransformerConfig(dim_model=380,dim_qkv=380,head_scale_size=8,feedforward_scale=2,dim_w_o_output_scaling=2)
        return BigHeadsTransformer(config)


# TODO add different recombination strategy for the heads such as addition for the head recombination
# instead of concatenation


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


class MixedActExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        dim_model = 512
        dim_feedforward = ((dim_model * 4) * 2) // 3
        config = TransformerConfig(dim_model=dim_model, dim_feedforward=dim_feedforward)
        return MixedActTransformer(config)


class MixedActSumOverMeanExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        dim_model = 512
        dim_feedforward = ((dim_model * 4) * 2) // 3
        config = TransformerConfig(dim_model=dim_model, dim_feedforward=dim_feedforward)
        return MixedActSumOverMeanTransformer(config)


class MixedActSOMDropoutExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        dim_model = 512
        dim_feedforward = ((dim_model * 4) * 2) // 3
        config = TransformerConfig(dim_model=dim_model, dim_feedforward=dim_feedforward)
        return MixedActSOMDropoutTransformer(config)


class NoSelfAttentionResidualExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return NoSelfAttentionResidualTransformer(config)
