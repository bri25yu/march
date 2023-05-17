from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig
from march.models.no_ff import NoFFTransformer
from march.models.values_relu import ValuesReluTransformer, ValuesReluFirstFFTransformer

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


class ValuesReluExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return ValuesReluTransformer(config)


class ValuesReluNoUpProjExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=960, feedforward_scale=1)
        return ValuesReluTransformer(config)
    

class ValuesReluFirstFFExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=1024)
        return ValuesReluFirstFFTransformer(config)
