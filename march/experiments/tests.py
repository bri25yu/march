from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig
from march.models.big_heads import BigHeadsTransformer, BigHeadsTransformerConfig
from march.models.TPWeights import TPWeightsTransformer
from march.models.big_heads_summed import BigHeadsSummedTransformerConfig, BigHeadsSummedTransformer
from march.models.no_ff import NoFFTransformer

from march.experiments.baseline import BaselineExperiment

from transformers import Seq2SeqTrainingArguments


class MoreHeadsLessQKVDimExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_qkv=TransformerConfig.dim_qkv // 2)
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


class TPWeightsExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=732,num_heads=12,dim_qkv=732//12)
        return TPWeightsTransformer(config)


class TPWeightsReduceLayersExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(num_layers=22)
        return TPWeightsTransformer(config)


class NoFFExperiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments["max_steps"] = 1000
        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return NoFFTransformer(config)


class NoFFParamMatchExperiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments["max_steps"] = 1000
        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=1088)
        return NoFFTransformer(config)
