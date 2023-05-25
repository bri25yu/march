from march.models.baseline import TransformerBase, TransformerConfig
from march.models.no_ff import NoFFTransformer
from march.models.values_relu import ValuesReluTransformer, ValuesReluFirstFFTransformer
from march.models.fixed_change_attention_based_summarization import FCABSTransformer, FCABSTransformerConfig

from march.experiments.baseline import BaselineExperiment


class NoFFExperiment(BaselineExperiment):
    NUM_STEPS = 1_000

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return NoFFTransformer(config)


class NoFFParamMatchExperiment(BaselineExperiment):
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


class FCABSExperiment(BaselineExperiment):
    NUM_STEPS = 1_000

    def get_model(self) -> TransformerBase:
        config = FCABSTransformerConfig()
        return FCABSTransformer(config)


class FCABSLdrop32Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = FCABSTransformerConfig(L_drop=32)
        return FCABSTransformer(config)
