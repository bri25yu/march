from march.models.TPEmbeddings import *

from march.experiments.baseline import *


class TPEmbeddingsBaseline1Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TPEmbeddingsConfig(num_layers=20,num_roles=89)
        return TPEmbeddingsBaselineTransformer(config)


class TPEmbeddingsBaseline2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TPEmbeddingsConfig(num_layers=16,num_roles=222)
        return TPEmbeddingsBaselineTransformer(config)


class TPEmbeddingsBaseline3Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TPEmbeddingsConfig(num_layers=12,num_roles=445)
        return TPEmbeddingsBaselineTransformer(config)


class TPEmbeddingsBaseline4Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TPEmbeddingsConfig(num_layers=6,num_roles=1335)
        return TPEmbeddingsBaselineTransformer(config)


class TPEmbeddingsBaseline3EncoderExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TPEmbeddingsConfig(num_layers=12,num_roles=445)
        return TPEmbeddingsBaselineEncoderTransformer(config)


class TPEmbeddingsBaseline3DecoderExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TPEmbeddingsConfig(num_layers=12,num_roles=445)
        return TPEmbeddingsBaselineDecoderTransformer(config)
