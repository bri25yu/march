from typing import Any, Dict

from abc import ABC, abstractmethod

from os.path import join

import json

from datasets import DatasetDict

from torch.cuda import device_count, get_device_properties

from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast, PrinterCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments

from march import CONFIG_DIR, RESULTS_DIR
from march.tokenization import EOS_TOKEN, load_tokenizer
from march.models.baseline import TransformerBase


class ExperimentBase(ABC):
    @abstractmethod
    def load_dataset_dict(self, tokenizer: PreTrainedTokenizerFast) -> DatasetDict:
        pass

    @abstractmethod
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        """
        Example:

        default_training_arguments = self.load_default_training_arguments()
        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)
        """
        pass

    @abstractmethod
    def get_model(self) -> TransformerBase:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def output_dir(self) -> str:
        return join(RESULTS_DIR, self.name)

    def load_default_training_arguments(self) -> Dict[str, Any]:
        args_dict = json.load(open(join(CONFIG_DIR, "default_training_arguments.json")))

        num_gpus = device_count()

        assert num_gpus > 0, "Training models is supposed to occur on GPUs, no GPUs found!"

        if num_gpus == 1:
            gpu_total_memory = get_device_properties(0).total_memory
            gpu_total_memory_gb = gpu_total_memory // (10 ** 9)
            assert gpu_total_memory_gb >= 42, f"Default batch size requires at least 40GB of GPU memory. The current GPU only has {gpu_total_memory_gb}GB."
        else:
            deepspeed_config = json.load(open(join(CONFIG_DIR, "deepspeed.json")))
            args_dict["deepspeed"] = deepspeed_config

            batch_size_per_gpu = args_dict["per_device_train_batch_size"]
            total_batch_size = batch_size_per_gpu * args_dict["gradient_accumulation_steps"]
            args_dict["gradient_accumulation_steps"] = total_batch_size // (batch_size_per_gpu * num_gpus)

            args_dict["bf16"] = True

        return args_dict

    def train(self) -> None:
        tokenizer = load_tokenizer()
        dataset_dict = self.load_dataset_dict(tokenizer)
        self._validate_dataset_dict(dataset_dict)
        model = self.get_model()
        print(f"Current model parameter count: {model.count_parameters():,}")

        base_data_collator = DataCollatorForSeq2Seq(tokenizer)
        bos_token_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
        def data_collator(examples):
            for example in examples:
                example["decoder_input_ids"] = [bos_token_id] + example["labels"][:-1]

            examples = base_data_collator(examples)

            return examples

        training_arguments = self.get_training_arguments()
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_arguments,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.remove_callback(PrinterCallback)

        trainer.train()

    def _validate_dataset_dict(self, dataset_dict: DatasetDict) -> None:
        expected_splits = ["train", "validation", "test"]
        actual_splits = list(dataset_dict)
        assert set(expected_splits) == set(actual_splits), f"Expected dataset dict to have splits {expected_splits}, but got {actual_splits}."

        expected_columns = ["input_ids", "labels"]
        actual_columns = dataset_dict["train"].column_names
        assert set(expected_columns) == set(actual_columns), f"Expected dataset to have columns {expected_columns}, but got {actual_columns}."
