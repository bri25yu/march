from march.experiments.baseline import *


class MoreHeadsMoreDimLessLayersExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()

        less_layers = 6
        extra_64 = 1
        extra_num_heads = 4

        config.num_layers = config.num_layers - less_layers

        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64 + extra_num_heads

        return BaselineTransformer(config)


class MoreHeadsMoreDimLessLayers2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()

        less_layers = 10
        extra_64 = 2
        extra_num_heads = 8

        config.num_layers = config.num_layers - less_layers

        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64 + extra_num_heads

        return BaselineTransformer(config)


class MoreHeadsMoreDimLessLayers3Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()

        less_layers = 14
        extra_64 = 3
        extra_num_heads = 15

        config.num_layers = config.num_layers - less_layers

        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64 + extra_num_heads

        return BaselineTransformer(config)


class MoreHeadsMoreDimLessLayers4Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()

        less_layers = 16
        extra_64 = 4
        extra_num_heads = 20

        config.num_layers = config.num_layers - less_layers

        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64 + extra_num_heads

        return BaselineTransformer(config)


class MoreHeadsMoreDimLessLayers5Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()

        less_layers = 20
        extra_64 = 8
        extra_num_heads = 38

        config.num_layers = config.num_layers - less_layers

        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64 + extra_num_heads

        return BaselineTransformer(config)


class MoreHeadsMoreDimLessLayers6Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()

        less_layers = 22
        extra_64 = 16
        extra_num_heads = 52

        config.num_layers = config.num_layers - less_layers

        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64 + extra_num_heads

        return BaselineTransformer(config)
