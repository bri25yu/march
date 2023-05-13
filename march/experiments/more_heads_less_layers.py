from march.experiments.baseline import *


class MoreHeadsLessLayersExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 2
        config.num_heads = config.num_heads + 2
        return BaselineTransformer(config)


class MoreHeadsLessLayers2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 4
        config.num_heads = config.num_heads + 4
        return BaselineTransformer(config)


class MoreHeadsLessLayers3Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 8
        config.num_heads = config.num_heads + 10
        return BaselineTransformer(config)


class MoreHeadsLessLayers4Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 12
        config.num_heads = config.num_heads + 19
        return BaselineTransformer(config)


class MoreHeadsLessLayers5Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 16
        config.num_heads = config.num_heads + 34
        return BaselineTransformer(config)


class MoreHeadsLessLayers6Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 20
        config.num_heads = config.num_heads + 66
        return BaselineTransformer(config)


class MoreHeadsLessLayers7Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 22
        config.num_heads = config.num_heads + 112
        return BaselineTransformer(config)
