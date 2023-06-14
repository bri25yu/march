from typing import Optional

from march.experiments.baseline_v2 import BaselineV2Config, BaselineV2Transformer, BaselineV2Experiment


# Baseline is 8 decoder layers / 896 = (768 + 2 * 64) model dim
class ModelDimLayersV2ExperimentBase(BaselineV2Experiment):
    NUM_DECODER_LAYERS: Optional[int] = None
    DIM_QKV_MUL: Optional[int] = None

    def get_model(self) -> BaselineV2Transformer:
        default_dim_model = BaselineV2Config.dim_model
        dim_qkv = BaselineV2Config.dim_qkv
        config = BaselineV2Config(
            dim_model=default_dim_model + self.DIM_QKV_MUL * dim_qkv,
            num_decoder_layers=self.NUM_DECODER_LAYERS,
        )
        return BaselineV2Transformer(config)


class ModelDim2304Layers1V2Experiment(ModelDimLayersV2ExperimentBase):
    NUM_DECODER_LAYERS = 1
    DIM_QKV_MUL = 22


class ModelDim1728Layers2V2Experiment(ModelDimLayersV2ExperimentBase):
    NUM_DECODER_LAYERS = 2
    DIM_QKV_MUL = 13


class ModelDim1408Layers3V2Experiment(ModelDimLayersV2ExperimentBase):
    NUM_DECODER_LAYERS = 3
    DIM_QKV_MUL = 8


class ModelDim1216Layers4V2Experiment(ModelDimLayersV2ExperimentBase):
    NUM_DECODER_LAYERS = 4
    DIM_QKV_MUL = 5


class ModelDim1152Layers5V2Experiment(ModelDimLayersV2ExperimentBase):
    NUM_DECODER_LAYERS = 5
    DIM_QKV_MUL = 4


class ModelDim1024Layers6V2Experiment(ModelDimLayersV2ExperimentBase):
    NUM_DECODER_LAYERS = 6
    DIM_QKV_MUL = 2


class ModelDim960Layers7V2Experiment(ModelDimLayersV2ExperimentBase):
    NUM_DECODER_LAYERS = 7
    DIM_QKV_MUL = 1


class ModelDim832Layers9V2Experiment(ModelDimLayersV2ExperimentBase):
    NUM_DECODER_LAYERS = 9
    DIM_QKV_MUL = -1


class ModelDim832Layers10V2Experiment(ModelDimLayersV2ExperimentBase):
    NUM_DECODER_LAYERS = 10
    DIM_QKV_MUL = -1


class ModelDim768Layers11V2Experiment(ModelDimLayersV2ExperimentBase):
    NUM_DECODER_LAYERS = 11
    DIM_QKV_MUL = -2


class ModelDim768Layers12V2Experiment(ModelDimLayersV2ExperimentBase):
    NUM_DECODER_LAYERS = 12
    DIM_QKV_MUL = -2


class ModelDim704Layers13V2Experiment(ModelDimLayersV2ExperimentBase):
    NUM_DECODER_LAYERS = 13
    DIM_QKV_MUL = -3


class ModelDim704Layers14V2Experiment(ModelDimLayersV2ExperimentBase):
    NUM_DECODER_LAYERS = 14
    DIM_QKV_MUL = -3


class ModelDim640Layers15V2Experiment(ModelDimLayersV2ExperimentBase):
    NUM_DECODER_LAYERS = 15
    DIM_QKV_MUL = -4
