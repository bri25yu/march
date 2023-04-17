from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig
from march.models.tests import ScalingHeadsTransformer
from march.experiments.baseline import BaselineWikiTextExperiment


class MoreHeadsLessLayersExperiment(BaselineWikiTextExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(num_layers=4, num_heads=16)
        return BaselineTransformer(config)


class MoreHeadsLessQKVDimExperiment(BaselineWikiTextExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_qkv=16)
        return BaselineTransformer(config)


class ScalingHeadsExperiment(BaselineWikiTextExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return ScalingHeadsTransformer(config)
