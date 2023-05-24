from march.experiments.baseline import BaselineExperiment, BaselineTransformer, TransformerBase, TransformerConfig


class ModelDim512Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=512)
        return BaselineTransformer(config)


class ModelDim1024Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=1024)
        return BaselineTransformer(config)


class Layers18Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(num_layers=18)
        return BaselineTransformer(config)


class Layers30Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(num_layers=30)
        return BaselineTransformer(config)


class FFDim2DExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(feedforward_scale=2.0)
        return BaselineTransformer(config)


class FFDim8DExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(feedforward_scale=8.0)
        return BaselineTransformer(config)


class ModelDim832Layers18Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        default_dim_model = TransformerConfig.dim_model
        dim_qkv = TransformerConfig.dim_qkv
        config = TransformerConfig(
            dim_model=default_dim_model + 1 * dim_qkv,
            num_layers=18,
        )
        return BaselineTransformer(config)


class ModelDim640Layers30Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        default_dim_model = TransformerConfig.dim_model
        dim_qkv = TransformerConfig.dim_qkv
        config = TransformerConfig(
            dim_model=default_dim_model - 2 * dim_qkv,
            num_layers=30,
        )
        return BaselineTransformer(config)


class ModelDim2304Layers2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        default_dim_model = TransformerConfig.dim_model
        dim_qkv = TransformerConfig.dim_qkv
        config = TransformerConfig(
            dim_model=default_dim_model + 24 * dim_qkv,
            num_layers=2,
        )
        return BaselineTransformer(config)
