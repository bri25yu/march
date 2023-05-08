from march.experiments.baseline import *


class MoreDimLessLayersExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 4

        extra_64 = 1
        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64
        return BaselineTransformer(config)


class MoreDimLessLayers2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 6

        extra_64 = 2
        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64
        return BaselineTransformer(config)


class MoreDimLessLayers3Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 10

        extra_64 = 4
        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64
        return BaselineTransformer(config)


class MoreDimLessLayers4Experiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_double_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 16

        extra_64 = 11
        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64
        return BaselineTransformer(config)


class MoreDimLessLayers5Experiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_double_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 20

        extra_64 = 20
        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64
        return BaselineTransformer(config)


class MoreDimLessLayers6Experiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_double_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 22

        extra_64 = 32
        config.dim_model = config.dim_model + 64 * extra_64
        config.num_heads = config.num_heads + extra_64
        return BaselineTransformer(config)
