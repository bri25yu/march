from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig
from march.models.no_ff import NoFFTransformer

from march.experiments.baseline import BaselineExperiment


class NoFFExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return NoFFTransformer(config)


class NoFFParamMatchExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=1088)
        return NoFFTransformer(config)
