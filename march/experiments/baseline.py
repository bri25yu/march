from datasets import DatasetDict

from transformers import PreTrainedTokenizerFast, Seq2SeqTrainingArguments

from march.datasets.wikipedia import load_wikipedia_baseline
from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig
from march.models.perfect_overfit import PerfectOverfitTransformer
from march.experiments.base import ExperimentBase


class BaselineExperiment(ExperimentBase):
    def load_dataset_dict(self, tokenizer: PreTrainedTokenizerFast) -> DatasetDict:
        return load_wikipedia_baseline()

    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return BaselineTransformer(config)


class PerfectOverfitExperiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()

        target_total_batch_size = 64 * 16
        train_batch_size = 64 * 16
        assert target_total_batch_size % train_batch_size == 0
        gradient_accumulation_steps = target_total_batch_size // train_batch_size

        default_training_arguments["per_device_train_batch_size"] = train_batch_size
        default_training_arguments["per_device_eval_batch_size"] = train_batch_size * 2
        default_training_arguments["gradient_accumulation_steps"] = gradient_accumulation_steps

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig(num_layers=2, dim_model=64, dropout_prob=0.0)
        return PerfectOverfitTransformer(config)


class BestExperiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()

        target_total_batch_size = 64 * 16
        train_batch_size = 4
        assert target_total_batch_size % train_batch_size == 0
        gradient_accumulation_steps = target_total_batch_size // train_batch_size

        default_training_arguments["per_device_train_batch_size"] = train_batch_size
        default_training_arguments["per_device_eval_batch_size"] = train_batch_size * 2
        default_training_arguments["gradient_accumulation_steps"] = gradient_accumulation_steps

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=1024, num_layers=48)  # Match t5-large
        return BaselineTransformer(config)
