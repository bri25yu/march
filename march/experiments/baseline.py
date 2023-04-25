from typing import Any, Dict

from types import MethodType

from datasets import DatasetDict

from transformers import PreTrainedTokenizerFast, Seq2SeqTrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

from march.datasets.wikipedia import load_wikipedia_baseline, load_wikipedia_baseline_t5
from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig
from march.experiments.base import ExperimentBase


def update_with_half_batch_size(training_arguments_dict: Dict[str, Any]) -> Dict[str, Any]:
    original_batch_size = training_arguments_dict["per_device_train_batch_size"]
    original_grad_accumulation = training_arguments_dict["gradient_accumulation_steps"]

    training_arguments_dict["per_device_train_batch_size"] = original_batch_size // 2
    training_arguments_dict["per_device_eval_batch_size"] = original_batch_size
    training_arguments_dict["gradient_accumulation_steps"] = original_grad_accumulation * 2

    return training_arguments_dict


class BaselineExperiment(ExperimentBase):
    def load_dataset_dict(self, tokenizer: PreTrainedTokenizerFast) -> DatasetDict:
        return load_wikipedia_baseline()

    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return BaselineTransformer(config)


class BaselineFP32Experiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()

        default_training_arguments["bf16"] = False
        default_training_arguments = update_with_half_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)


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


class BaselineT5Experiment(BaselineExperiment):
    MODEL_NAME = "t5-base"

    def load_default_tokenizer(self) -> PreTrainedTokenizerFast:
        return AutoTokenizer.from_pretrained(self.MODEL_NAME, model_max_length=1024)

    def load_dataset_dict(self, tokenizer: PreTrainedTokenizerFast) -> DatasetDict:
        return load_wikipedia_baseline_t5()

    def get_model(self) -> TransformerBase:
        config = AutoConfig.from_pretrained(self.MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_config(config)
        setattr(model, "count_parameters", MethodType(BaselineTransformer.count_parameters, model))

        return model
