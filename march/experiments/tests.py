from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig
from march.models.no_ff import NoFFTransformer

from march.experiments.baseline import BaselineExperiment

from transformers import Seq2SeqTrainingArguments


class NoFFExperiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments["max_steps"] = 1000
        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return NoFFTransformer(config)


class NoFFParamMatchExperiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments["max_steps"] = 1000
        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=1088)
        return NoFFTransformer(config)
