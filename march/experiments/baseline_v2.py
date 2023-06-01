from march.models.baseline_v2 import BaselineV2Config, BaselineV2Transformer
from march.experiments.baseline import BaselineExperiment, TransformerBase


class BaselineV2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = BaselineV2Config()
        return BaselineV2Transformer(config)
