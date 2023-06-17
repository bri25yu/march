from typing import Optional

from march.experiments.baseline_v2 import BaselineV2Config, BaselineV2Transformer, BaselineV2Experiment


# Baseline is 2368 = 896 * (2 / 3) * 4 dim_ff / 16 layers
class FFScaleLayersV2ExperimentBase(BaselineV2Experiment):
    FEEDFORWARD_SCALE: Optional[float] = None
    NUM_DECODER_LAYERS: Optional[int] = None

    def get_model(self) -> BaselineV2Transformer:
        config = BaselineV2Config(
            feedforward_scale=self.FEEDFORWARD_SCALE,
            num_decoder_layers=self.NUM_DECODER_LAYERS,
        )
        return BaselineV2Transformer(config)


class FFScale38Layers1Experiment(FFScaleLayersV2ExperimentBase):
    FEEDFORWARD_SCALE = 38
    NUM_DECODER_LAYERS = 1


class FFScale18Layers2Experiment(FFScaleLayersV2ExperimentBase):
    FEEDFORWARD_SCALE = 18
    NUM_DECODER_LAYERS = 2


class FFScale11Layers3Experiment(FFScaleLayersV2ExperimentBase):
    FEEDFORWARD_SCALE = 11
    NUM_DECODER_LAYERS = 3


class FFScale8Layers4Experiment(FFScaleLayersV2ExperimentBase):
    FEEDFORWARD_SCALE = 8
    NUM_DECODER_LAYERS = 4


class FFScale6Layers5Experiment(FFScaleLayersV2ExperimentBase):
    FEEDFORWARD_SCALE = 6
    NUM_DECODER_LAYERS = 5


class FFScale4_5Layers6Experiment(FFScaleLayersV2ExperimentBase):
    FEEDFORWARD_SCALE = 4.5
    NUM_DECODER_LAYERS = 6


class FFScale3_5Layers7Experiment(FFScaleLayersV2ExperimentBase):
    FEEDFORWARD_SCALE = 3.5
    NUM_DECODER_LAYERS = 7


class FFScale3Layers8Experiment(FFScaleLayersV2ExperimentBase):
    FEEDFORWARD_SCALE = 3
    NUM_DECODER_LAYERS = 8


class FFScale2_5Layers9Experiment(FFScaleLayersV2ExperimentBase):
    FEEDFORWARD_SCALE = 2.5
    NUM_DECODER_LAYERS = 9


class FFScale2Layers10Experiment(FFScaleLayersV2ExperimentBase):
    FEEDFORWARD_SCALE = 2
    NUM_DECODER_LAYERS = 10


class FFScale1_5Layers11Experiment(FFScaleLayersV2ExperimentBase):
    FEEDFORWARD_SCALE = 1.5
    NUM_DECODER_LAYERS = 11


class FFScale1_25Layers12Experiment(FFScaleLayersV2ExperimentBase):
    FEEDFORWARD_SCALE = 1.25
    NUM_DECODER_LAYERS = 12


class FFScale1Layers13Experiment(FFScaleLayersV2ExperimentBase):
    FEEDFORWARD_SCALE = 1
    NUM_DECODER_LAYERS = 13


class FFScale0_75Layers14Experiment(FFScaleLayersV2ExperimentBase):
    FEEDFORWARD_SCALE = 0.75
    NUM_DECODER_LAYERS = 14


class FFScale0_5Layers15Experiment(FFScaleLayersV2ExperimentBase):
    FEEDFORWARD_SCALE = 0.5
    NUM_DECODER_LAYERS = 15
