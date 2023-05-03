from march.experiments.baseline import *


class FFDimHalfToLayersExperiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_half_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig(feedforward_scale=0.5)
        config.num_layers = config.num_layers + 24

        return BaselineTransformer(config)


class FFDimHalfToDimExperiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_half_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        base_dim = TransformerConfig.dim_model
        config = TransformerConfig(feedforward_scale=0.5, dim_model=base_dim + 64 * 4)

        return BaselineTransformer(config)


class FFDimSameToLayersExperiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_half_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig(feedforward_scale=1.0)
        config.num_layers = config.num_layers + 18

        return BaselineTransformer(config)


class FFDimSameToDimExperiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_half_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        base_dim = TransformerConfig.dim_model
        config = TransformerConfig(feedforward_scale=1.0, dim_model=base_dim + 64 * 3)

        return BaselineTransformer(config)


class FFDimDoubleToLayersExperiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_half_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig(feedforward_scale=2.0)
        config.num_layers = config.num_layers + 8

        return BaselineTransformer(config)


class FFDimDoubleToDimExperiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_half_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        base_dim = TransformerConfig.dim_model
        config = TransformerConfig(feedforward_scale=2.0, dim_model=base_dim + 64 * 2)

        return BaselineTransformer(config)


class FFDimOctupleFromLayersExperiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_half_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig(feedforward_scale=8.0)
        config.num_layers = config.num_layers - 10

        return BaselineTransformer(config)


class FFDimOctupleFromDimExperiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_half_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        base_dim = TransformerConfig.dim_model
        config = TransformerConfig(feedforward_scale=8.0, dim_model=base_dim - 64 * 3)
        config.num_layers = config.num_layers + 4

        return BaselineTransformer(config)
