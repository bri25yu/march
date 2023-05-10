from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig

from march.experiments.baseline import BaselineExperiment


# Trying out baseline model but with less params in different places, how does this affect performance? 


class LessParamsLessLayers1Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(num_layers=18)
        return BaselineTransformer(config)


class LessParamsLessLayers2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(num_layers=12)
        return BaselineTransformer(config)


class LessParamsLessLayers3Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(num_layers=6)
        return BaselineTransformer(config)


class LessParamsLessLayers4Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(num_layers=2)
        return BaselineTransformer(config)


class LessParamsLessD_Model1Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=576)
        return BaselineTransformer(config)


class LessParamsLessD_Model2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=384)
        return BaselineTransformer(config)
        

class LessParamsLessD_ModelSameHeads1Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=576, dim_qkv=48)
        return BaselineTransformer(config)
        

class LessParamsLessD_ModelSameHeads2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=384, dim_qkv=32)
        return BaselineTransformer(config)
        