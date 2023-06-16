from typing import Optional

from march.experiments.baseline_v2 import BaselineV2Config, BaselineV2Transformer, BaselineV2Experiment


# Baseline is 896 = (768 + 2 * 64) model dim / 2368 = 896 * (2 / 3) * 4
class ModelDimFFScaleV2ExperimentBase(BaselineV2Experiment):
    FEEDFORWARD_SCALE: Optional[float] = None
    DIM_QKV_MUL: Optional[int] = None

    def get_model(self) -> BaselineV2Transformer:
        default_dim_model = BaselineV2Config.dim_model
        dim_qkv = BaselineV2Config.dim_qkv
        config = BaselineV2Config(
            dim_model=default_dim_model + self.DIM_QKV_MUL * dim_qkv,
            feedforward_scale=self.FEEDFORWARD_SCALE,
        )
        return BaselineV2Transformer(config)


class ModelDim1280FFScaleQuarterV2Experiment(ModelDimFFScaleV2ExperimentBase):
    FEEDFORWARD_SCALE = 0.25
    DIM_QKV_MUL = 6


class ModelDim1216FFScaleHalfV2Experiment(ModelDimFFScaleV2ExperimentBase):
    FEEDFORWARD_SCALE = 0.5
    DIM_QKV_MUL = 5


class ModelDim1088FFScale1V2Experiment(ModelDimFFScaleV2ExperimentBase):
    FEEDFORWARD_SCALE = 1.0
    DIM_QKV_MUL = 3


class ModelDim960FFScale2V2Experiment(ModelDimFFScaleV2ExperimentBase):
    FEEDFORWARD_SCALE = 2.0
    DIM_QKV_MUL = 1


class ModelDim832FFScale4V2Experiment(ModelDimFFScaleV2ExperimentBase):
    FEEDFORWARD_SCALE = 4.0
    DIM_QKV_MUL = -1


class ModelDim640FFScale8V2Experiment(ModelDimFFScaleV2ExperimentBase):
    FEEDFORWARD_SCALE = 8.0
    DIM_QKV_MUL = -4
