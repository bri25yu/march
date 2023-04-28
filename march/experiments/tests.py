from transformers import Seq2SeqTrainingArguments

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
from march.models.no_key_value_weights_cross_attn import NoKeyValueWeightsCrossAttentionTransformer
from march.models.sparse_seqlen_attention import NoSelfAttentionResidualTransformer
from march.models.speedups import FastTransformer
from march.models.TPWeights import TPWeightsTransformer
from march.models.big_heads_summed import BigHeadsSummedTransformerConfig, BigHeadsSummedTransformer

from march.experiments.baseline import BaselineExperiment, update_with_double_batch_size, update_with_half_batch_size


class MoreHeadsLessQKVDimExperiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_half_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_qkv=TransformerConfig.dim_qkv // 2)
        return BaselineTransformer(config)


class MoreHeadsLessLayersExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 2
        config.num_heads = config.num_heads + 2
        return BaselineTransformer(config)


class MoreHeadsLessLayers2Experiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_half_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 4
        config.num_heads = config.num_heads + 4
        return BaselineTransformer(config)


class MoreHeadsLessLayers3Experiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_half_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 8
        config.num_heads = config.num_heads + 10
        return BaselineTransformer(config)


class MoreHeadsLessLayers4Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 12
        config.num_heads = config.num_heads + 19
        return BaselineTransformer(config)


class MoreHeadsLessLayers5Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 16
        config.num_heads = config.num_heads + 34
        return BaselineTransformer(config)


class MoreHeadsLessLayers6Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 20
        config.num_heads = config.num_heads + 66
        return BaselineTransformer(config)


class MoreHeadsLessLayersNoKVExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 2
        config.num_heads = config.num_heads + 5
        return NoKeyValueWeightsCrossAttentionTransformer(config)


class MoreHeadsLessLayersNoKV2Experiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_half_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 4
        config.num_heads = config.num_heads + 9
        return NoKeyValueWeightsCrossAttentionTransformer(config)


class MoreHeadsLessLayersNoKV3Experiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_half_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 8
        config.num_heads = config.num_heads + 19
        return NoKeyValueWeightsCrossAttentionTransformer(config)


class MoreHeadsLessLayersNoKV4Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 12
        config.num_heads = config.num_heads + 36
        return NoKeyValueWeightsCrossAttentionTransformer(config)


class MoreHeadsLessLayersNoKV5Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 16
        config.num_heads = config.num_heads + 69
        return NoKeyValueWeightsCrossAttentionTransformer(config)


class MoreHeadsLessLayersNoKV6Experiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_half_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 20
        config.num_heads = config.num_heads + 170
        return NoKeyValueWeightsCrossAttentionTransformer(config)


class MoreDimLessLayersExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 4

        extra_64 = 1
        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64
        return BaselineTransformer(config)


class MoreDimLessLayers2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 6

        extra_64 = 2
        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64
        return BaselineTransformer(config)


class MoreDimLessLayers3Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 10

        extra_64 = 4
        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64
        return BaselineTransformer(config)


class MoreDimLessLayers4Experiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_double_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 16

        extra_64 = 11
        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64
        return BaselineTransformer(config)


class MoreDimLessLayers5Experiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_double_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 20

        extra_64 = 20
        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64
        return BaselineTransformer(config)


class MoreHeadsMoreDimLessLayersExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()

        less_layers = 6
        extra_64 = 1
        extra_num_heads = 4

        config.num_layers = config.num_layers - less_layers

        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64 + extra_num_heads

        return BaselineTransformer(config)


class MoreHeadsMoreDimLessLayers2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()

        less_layers = 10
        extra_64 = 2
        extra_num_heads = 8

        config.num_layers = config.num_layers - less_layers

        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64 + extra_num_heads

        return BaselineTransformer(config)


class MoreHeadsMoreDimLessLayers3Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()

        less_layers = 14
        extra_64 = 3
        extra_num_heads = 14

        config.num_layers = config.num_layers - less_layers

        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64 + extra_num_heads

        return BaselineTransformer(config)


class MoreHeadsMoreDimLessLayers4Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()

        less_layers = 16
        extra_64 = 4
        extra_num_heads = 20

        config.num_layers = config.num_layers - less_layers

        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64 + extra_num_heads

        return BaselineTransformer(config)


class MoreHeadsMoreDimLessLayers5Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()

        less_layers = 20
        extra_64 = 8
        extra_num_heads = 38

        config.num_layers = config.num_layers - less_layers

        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64 + extra_num_heads

        return BaselineTransformer(config)


class MoreHeadsLessQKVDimLessLayersExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.dim_qkv = config.dim_qkv // 2
        config.num_layers = (config.num_layers * 3) // 4
        config.num_heads = config.num_heads * 4
        return BaselineTransformer(config)


class LessHeadsMoreQKVDimExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_qkv=TransformerConfig.dim_qkv * 2)
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
        config.num_heads = config.num_heads - 4
        return DatabaseTransformer(config)


class DatabaseFromLayersExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = DatabaseTransformerConfig()
        config.num_layers -= 6

        return DatabaseTransformer(config)


class DatabaseFromDimExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = DatabaseTransformerConfig()
        config.dim_model -= 64 * 3

        return DatabaseTransformer(config)


class APESumOverAverageExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return APESumOverAverageTransformer(config)


class APEUnitVarianceExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return APEUnitVarianceTransformer(config)


# Targeted experiments for BigHeadsTransformer:
# 2x d_kv for each head but smaller overall hidden dimension 
# size due to constraint of keeping the model size the same
# Head size D_kv = D_kv_orig x 2 = D_model / 6, scaling down D_model to 648
# w_o is from D_kv_orig x 2 x 12 -> D_model
# AKA w_o is from D_model x 2 -> D_model
# FF layer D_model -> 4 x D_model -> D_model
class BigHeadsExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsTransformerConfig(dim_model=636,head_scale_size=2)
        return BaselineTransformer(config)


# Same as above but reducing number of layers instead of hidden dimension
class BigHeadsReduceLayersExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsTransformerConfig(num_layers=16,head_scale_size=2)
        return BaselineTransformer(config)


# Same as above but reducing number of heads in addition to hidden dim
class BigHeadsReduceNumHeadsExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsTransformerConfig(dim_model=636,num_heads=8,head_scale_size=2)
        return BaselineTransformer(config)


class BigHeadsSummedExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsSummedTransformerConfig(dim_model=636,head_scale_size=2)
        return BigHeadsSummedTransformer(config)


# Head size D_kv = D_kv_orig x 6 = D_model / 2, scaling down D_model to 444
# w_o is from D_kv_orig x 6 x 12 -> D_model
# AKA w_o is from D_model x 6 -> D_model
# FF layer D_model -> 4 x D_model -> D_model
class BigHeads2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsTransformerConfig(dim_model=444,head_scale_size=6)
        return BaselineTransformer(config)


class BigHeads2SummedExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsSummedTransformerConfig(dim_model=444,head_scale_size=6)
        return BigHeadsSummedTransformer(config)


# Head size D_kv = D_kv_orig x 12 = D_model, scaling down D_model to 332
# w_o is from D_kv_orig x 12 x 12 -> D_model
# AKA w_o is from D_model x 12 -> D_model
# FF layer D_model -> 4 x D_model -> D_model
class BigHeads3Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsTransformerConfig(dim_model=332,head_scale_size=12)
        return BaselineTransformer(config)


class BigHeads3SummedExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsSummedTransformerConfig(dim_model=332,head_scale_size=12)
        return BigHeadsSummedTransformer(config)


# Head size D_kv = D_kv_orig x 2 = D_model / 6, scaling down D_model to 564
# w_o is now from D_kv_orig x 2 x 12 -> D_kv_orig x 2 x 12
# AKA w_o is from D_model x 2 -> D_model x 2
# FF layer D_model x 2 -> 4 x D_model -> D_model
class BigHeadsLinearW_oExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsTransformerConfig(dim_model=564,head_scale_size=2)
        return BigHeadsTransformer(config)


# Head size D_kv = D_kv_orig x 6 = D_model / 2, scaling down D_model to 288
# w_o is now from D_kv_orig x 6 x 12 -> D_kv_orig x 6 x 12
# AKA w_o is from D_model x 6 -> D_model x 6
# FF layer 6 x D_model -> 4 x D_model -> D_model
class BigHeadsLinearW_o2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsTransformerConfig(dim_model=288,head_scale_size=6)
        return BigHeadsTransformer(config)


# Head size D_kv = D_kv_orig x 12 = D_model, scaling down D_model to 168
# w_o is now from D_kv_orig x 12 x 12 -> D_kv_orig x 12 x 12
# AKA w_o is from D_model x 12 -> D_model x 12
# FF layer 12 x D_model -> 4 x D_model -> D_model
class BigHeadsLinearW_o3Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsTransformerConfig(dim_model=168,head_scale_size=12)
        return BigHeadsTransformer(config)


# Head size D_kv = D_kv_orig x 6 = D_model / 2, scaling down D_model to 306
# w_o is now from D_kv_orig x 6 x 12 -> D_kv_orig x 6 x 12
# AKA w_o is from D_model x 6 -> D_model x 4
# FF layer 4 x D_model -> 2 x D_model -> D_model
class BigHeadsDownProjectExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsTransformerConfig(dim_model=306,head_scale_size=6,feedforward_scale=2,dim_w_o_output_scaling=1.5)
        return BigHeadsTransformer(config)


# Head size D_kv = D_kv_orig x 12 = D_model, scaling down D_model to 176
# w_o is now from D_kv_orig x 12 x 12 -> D_kv_orig x 12 x 12
# AKA w_o is from D_model x 12 -> D_model x 4
# FF layer 4 x D_model -> 2 x D_model -> D_model
class BigHeadsDownProject2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BigHeadsTransformerConfig(dim_model=176,head_scale_size=12,feedforward_scale=2,dim_w_o_output_scaling=3)
        return BigHeadsTransformer(config)


# TODO
class BigHeadsNoW_oExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        # config = BigHeadsTransformerConfig(dim_model=448,dim_qkv=112,head_scale_size=2,use_w_o=False)
        # return BigHeadsTransformer(config)
        pass


# TODO add different recombination strategy for the heads such as addition for the head recombination
# instead of concatenation

# Add relu to the w_o output

# TODO Big heads reduce the number of heads instead of hidden dimensions
# TODO Big heads reduce the number 


class ReLUGatedLinearUnitExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        feedforward_scale = 4 * 2 / 3
        config = GatedLinearUnitTransformerConfig(
            feedforward_scale=feedforward_scale, gate_fn=GateFunctions.RELU
        )
        return GatedLinearUnitTransformer(config)


class GELUGatedLinearUnitExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        feedforward_scale = 4 * 2 / 3
        config = GatedLinearUnitTransformerConfig(
            feedforward_scale=feedforward_scale, gate_fn=GateFunctions.GELU
        )
        return GatedLinearUnitTransformer(config)


class SiLUGatedLinearUnitExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        feedforward_scale = 4 * 2 / 3
        config = GatedLinearUnitTransformerConfig(
            feedforward_scale=feedforward_scale, gate_fn=GateFunctions.SILU
        )
        return GatedLinearUnitTransformer(config)


class MixedActExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        feedforward_scale = 4 * 2 / 3
        config = TransformerConfig(feedforward_scale=feedforward_scale)
        return MixedActTransformer(config)


class MixedActSumOverMeanExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        feedforward_scale = 4 * 2 / 3
        config = TransformerConfig(feedforward_scale=feedforward_scale)
        return MixedActSumOverMeanTransformer(config)


class MixedActSOMDropoutExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        feedforward_scale = 4 * 2 / 3
        config = TransformerConfig(feedforward_scale=feedforward_scale)
        return MixedActSOMDropoutTransformer(config)


class NoSelfAttentionResidualExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return NoSelfAttentionResidualTransformer(config)


class FastTransformerExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return FastTransformer(config)


# Remove w_k and w_v from cross attention in the decoder
class NoKeyValueWeightsCrossAttentionExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return NoKeyValueWeightsCrossAttentionTransformer(config)


class NoKeyValueWeightsCrossAttentionWithExtraHeadsExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_heads += 2
        return NoKeyValueWeightsCrossAttentionTransformer(config)


class NoKeyValueWeightsCrossAttentionWithExtraDimExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.dim_model += config.num_heads * 2
        config.dim_qkv += 2
        return NoKeyValueWeightsCrossAttentionTransformer(config)


class TPWeightsExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=732,num_heads=12,dim_qkv=732//12)
        return TPWeightsTransformer(config)


class TPWeightsReduceLayersExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(num_layers=22)
        return TPWeightsTransformer(config)
