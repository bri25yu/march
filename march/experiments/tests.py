from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig
from march.experiments.baseline import BaselineWikiTextExperiment


class MoreHeadsLessLayersExperiment(BaselineWikiTextExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(num_layers=4, num_heads=32)
        return BaselineTransformer(config)
