from march.experiments.baseline import *

from march.models.mixed_act import (
    GatedLinearUnitTransformer,
    GatedLinearUnitTransformerConfig,
    GateFunctions,
)


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
