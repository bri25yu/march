from typing import Any, Dict

from abc import ABC, abstractmethod

from os import environ
from os.path import join

import json

from datasets import DatasetDict

from numpy.random import seed as set_numpy_seed

from torch.cuda import device_count
from torch import manual_seed as set_torch_seed

from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast, PrinterCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments

from march import CONFIG_DIR, RESULTS_DIR
from march.tokenization import EOS_TOKEN, load_tokenizer
from march.models.baseline import TransformerBase, LayerNorm, AttentionBase, AbsolutePositionEncoding


class CustomLoggingSeq2SeqTrainer(Seq2SeqTrainer):
    def log(self, logs: Dict[str, float]) -> None:
        if "LOG_WEIGHTS" in environ:
            modules_by_cls = lambda cls: [module for module in self.model.modules() if isinstance(module, cls)]

            layernorm_modules = modules_by_cls(LayerNorm)
            logs["layernorm_mean"] = sum([m.weight.data for m in layernorm_modules]).mean().item() / len(layernorm_modules)

            attention_modules = modules_by_cls(AttentionBase)
            logs["attention_w_q_mean"] = sum([m.w_q.weight.data for m in attention_modules]).mean().item() / len(attention_modules)

            logs["position_encoding_mean"] = self.model.position_encoding.timing_table.data.mean().item()
            logs["embedding_mean"] = self.model.embedding.weight.data.mean().item()

        return super().log(logs)


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
        assert num_gpus == 8, "Training models is supposed to occur on a DGX node of 8 GPUs!"

        deepspeed_config = json.load(open(join(CONFIG_DIR, "deepspeed.json")))
        args_dict["deepspeed"] = deepspeed_config

        return args_dict

    def train(self) -> None:
        training_arguments = self.get_training_arguments()

        set_numpy_seed(training_arguments.seed)
        set_torch_seed(training_arguments.seed)

        tokenizer = load_tokenizer()
        dataset_dict = self.load_dataset_dict(tokenizer)
        self._validate_dataset_dict(dataset_dict)
        model = self.get_model()

        base_data_collator = DataCollatorForSeq2Seq(tokenizer)
        bos_token_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
        def data_collator(examples):
            for example in examples:
                example["decoder_input_ids"] = [bos_token_id] + example["labels"][:-1]

            examples = base_data_collator(examples)

            return examples

        trainer = CustomLoggingSeq2SeqTrainer(
            model=model,
            args=training_arguments,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.remove_callback(PrinterCallback)
        trainer.log({"num_params": model.count_parameters()})

        trainer.train()

    def _validate_dataset_dict(self, dataset_dict: DatasetDict) -> None:
        expected_splits = ["train", "validation", "test"]
        actual_splits = list(dataset_dict)
        assert set(expected_splits) == set(actual_splits), f"Expected dataset dict to have splits {expected_splits}, but got {actual_splits}."

        expected_columns = ["input_ids", "labels"]
        actual_columns = dataset_dict["train"].column_names
        assert set(expected_columns) == set(actual_columns), f"Expected dataset to have columns {expected_columns}, but got {actual_columns}."
