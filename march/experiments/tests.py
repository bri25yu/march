from march.models.baseline import TransformerBase, TransformerConfig
from march.models.no_ff import NoFFTransformer
from march.models.values_relu import ValuesReluTransformer, ValuesReluFirstFFTransformer

from march.experiments.baseline import BaselineExperiment


class NoFFExperiment(BaselineExperiment):
    NUM_STEPS = 1_000

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return NoFFTransformer(config)


class NoFFParamMatchExperiment(BaselineExperiment):
    NUM_STEPS = 1_000

    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=1088)
        return NoFFTransformer(config)


class ValuesReluExperiment(BaselineExperiment):
    NUM_STEPS = 1_000

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return ValuesReluTransformer(config)


class ValuesReluNoUpProjExperiment(BaselineExperiment):
    NUM_STEPS = 1_000

    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=960, feedforward_scale=1)
        return ValuesReluTransformer(config)


class ValuesReluFirstFFExperiment(BaselineExperiment):
    NUM_STEPS = 1_000

    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=1024)
        return ValuesReluFirstFFTransformer(config)
