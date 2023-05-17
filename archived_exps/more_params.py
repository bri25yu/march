from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig

from march.experiments.baseline import BaselineExperiment


# Trying out baseline model but with more params in different places, how does this affect performance? 


class MoreParamsMoreLayers1Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(num_layers=30)
        return BaselineTransformer(config)


class MoreParamsMoreLayers2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(num_layers=36)
        return BaselineTransformer(config)


class MoreParamsMoreD_model1Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=960)
        return BaselineTransformer(config)


class MoreParamsMoreD_model2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=1152)
        return BaselineTransformer(config)


class MoreParamsMoreD_modelSameHeads1Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=960, dim_qkv=80)
        return BaselineTransformer(config)


class MoreParamsMoreD_modelSameHeads2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=1152, dim_qkv=96)
        return BaselineTransformer(config)
