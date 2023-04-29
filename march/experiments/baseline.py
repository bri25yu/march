from typing import Any, Dict

from types import MethodType

from datasets import DatasetDict

from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast, Seq2SeqTrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

from march.datasets.wikipedia import load_wikipedia_baseline, load_wikipedia_baseline_t5
from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig
from march.experiments.base import ExperimentBase


def update_with_double_batch_size(training_arguments_dict: Dict[str, Any]) -> Dict[str, Any]:
    original_batch_size = training_arguments_dict["per_device_train_batch_size"]
    original_grad_accumulation = training_arguments_dict["gradient_accumulation_steps"]

    training_arguments_dict["per_device_train_batch_size"] = original_batch_size * 2
    training_arguments_dict["per_device_eval_batch_size"] = original_batch_size
    training_arguments_dict["gradient_accumulation_steps"] = original_grad_accumulation // 2

    return training_arguments_dict


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

    def get_data_collator(self, tokenizer: PreTrainedTokenizerFast):
        # Invert the mask for T5, change the pad token id

        base_data_collator = DataCollatorForSeq2Seq(tokenizer)
        bos_token_id = pad_token_id = tokenizer.pad_token_id
        def data_collator(examples):
            for example in examples:
                example["decoder_input_ids"] = [bos_token_id] + example["labels"][:-1]

            examples = base_data_collator(examples)

            # 0 for should mask, 1 otherwise
            examples["attention_mask"] = examples["input_ids"] != pad_token_id

            return examples

        return data_collator


class BaselineT5v1_1Experiment(BaselineT5Experiment):
    MODEL_NAME = "google/t5-v1_1-base"

    def get_model(self) -> TransformerBase:
        config = AutoConfig.from_pretrained(self.MODEL_NAME, dropout_rate=0.0)
        model = AutoModelForSeq2SeqLM.from_config(config)
        setattr(model, "count_parameters", MethodType(BaselineTransformer.count_parameters, model))

        return model


def modify_training_args_for_large_exp(training_arguments: Dict[str, Any]) -> Dict[str, Any]:
    initial_batch_size = training_arguments["per_device_train_batch_size"]
    initial_ga_steps = training_arguments["gradient_accumulation_steps"]
    initial_examples_per_gpu = initial_batch_size * initial_ga_steps

    new_examples_per_gpu = initial_examples_per_gpu * 4  # 1k steps is 4B tokens
    new_batch_size = initial_batch_size // 4  # Lower the batch size for larger model
    new_ga_steps = new_examples_per_gpu // 4

    training_arguments.update({
        "per_device_train_batch_size": new_batch_size,
        "per_device_eval_batch_size": new_batch_size * 2,
        "gradient_accumulation_steps": new_ga_steps,
    })

    return training_arguments


class BaselineT5LargeExperiment(BaselineT5Experiment):
    MODEL_NAME = "t5-large"

    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        training_arguments = self.load_default_training_arguments()
        training_arguments = modify_training_args_for_large_exp(training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **training_arguments)


class BaselineLargeExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(
            dim_model=1024,
            num_layers=48,
        )
        return BaselineTransformer(config)

    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        training_arguments = self.load_default_training_arguments()
        training_arguments = modify_training_args_for_large_exp(training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **training_arguments)


class BaselineT5v1_1LargeExperiment(BaselineT5v1_1Experiment):
    MODEL_NAME = "google/t5-v1_1-large"

    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        training_arguments = self.load_default_training_arguments()
        training_arguments = modify_training_args_for_large_exp(training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **training_arguments)
