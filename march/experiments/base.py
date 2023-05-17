from typing import Any, Dict

from abc import ABC, abstractmethod

from os.path import join

import json

from datasets import DatasetDict

from numpy.random import seed as set_numpy_seed

from torch.cuda import device_count
from torch import manual_seed as set_torch_seed
from torch.nn import Module

from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast, PrinterCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments

from march import CONFIG_DIR, RESULTS_DIR
from march.datasets.c4 import EOS_TOKEN, load_c4_tokenizer
from march.models.baseline import TransformerBase, LayerNorm, AttentionBase


class CustomLoggingSeq2SeqTrainer(Seq2SeqTrainer):
    def log(self, logs: Dict[str, float]) -> None:
        modules_by_cls = lambda cls: [module for module in self.model.modules() if isinstance(module, cls)]

        layernorm_modules = modules_by_cls(LayerNorm)
        if len(layernorm_modules) > 0:
            logs["layernorm_mean"] = sum([m.weight.data.mean() for m in layernorm_modules]).item() / len(layernorm_modules)
            logs["layernorm_max"] = sum([m.weight.data.max() for m in layernorm_modules]).item() / len(layernorm_modules)
            logs["layernorm_min"] = sum([m.weight.data.min() for m in layernorm_modules]).item() / len(layernorm_modules)

        attention_modules = modules_by_cls(AttentionBase)
        def get_attn_weight_mean(weight_name: str) -> float:
            weights = [getattr(m, weight_name).weight.data.mean() for m in attention_modules if hasattr(m, weight_name)]
            if len(weights) > 0:
                return sum(weights).item() / len(attention_modules)
            return 0.0

        if len(attention_modules) > 0:
            logs["attention_w_q_mean"] = get_attn_weight_mean("w_q")
            logs["attention_w_k_mean"] = get_attn_weight_mean("w_k")
            logs["attention_w_v_mean"] = get_attn_weight_mean("w_v")
            logs["attention_w_o_mean"] = get_attn_weight_mean("w_o")

        if hasattr(self.model, "position_encoding"):
            logs["position_encoding_mean"] = self.model.position_encoding.timing_table.data.mean().item()

        if hasattr(self.model, "embedding"):
            logs["embedding_mean"] = self.model.embedding.weight.data.mean().item()

        return super().log(logs)


class ExperimentBase(ABC):
    def __init__(self, batch_size_pow_scale: int=0) -> None:
        super().__init__()

        # For self.load_default_training_arguments
        self.batch_size_pow_scale = batch_size_pow_scale

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

        # Validate scale factor
        pow_scale = self.batch_size_pow_scale
        assert isinstance(pow_scale, int), f"The batch size scaling factor must be an integer, but got {pow_scale}"

        original_batch_size = args_dict["per_device_train_batch_size"]
        original_grad_accumulation = args_dict["gradient_accumulation_steps"]

        if pow_scale >= 0:  # Inc batch size, dec grad accumulation
            scale = int(pow(2, pow_scale))
            assert original_grad_accumulation % scale == 0, f"Power scale factor of 2 ** {pow_scale} = {scale} is too powerful for the current gradient accumulation steps {original_grad_accumulation}"
            new_batch_size = original_batch_size * scale
            new_grad_accumulation = original_grad_accumulation // scale
        else:  # Dec batch size, inc grad accumulation
            inv_scale = int(pow(2, -pow_scale))
            assert original_batch_size % inv_scale == 0, f"Power scale factor of 1 / 2 ** {-pow_scale} = 1 / {inv_scale} is too powerful for the current batch size {original_batch_size}"
            new_batch_size = original_batch_size // inv_scale
            new_grad_accumulation = original_grad_accumulation * inv_scale

        args_dict["per_device_train_batch_size"] = new_batch_size
        args_dict["per_device_eval_batch_size"] = 2 * new_batch_size
        args_dict["gradient_accumulation_steps"] = new_grad_accumulation

        return args_dict

    def load_default_tokenizer(self) -> PreTrainedTokenizerFast:
        return load_c4_tokenizer()

    def get_data_collator(self, tokenizer: PreTrainedTokenizerFast):
        base_data_collator = DataCollatorForSeq2Seq(tokenizer)
        bos_token_id = pad_token_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
        def data_collator(examples):
            for example in examples:
                example["decoder_input_ids"] = [bos_token_id] + example["labels"][:-1]

            examples = base_data_collator(examples)

            # 1 for should mask, 0 otherwise
            examples["attention_mask"] = examples["input_ids"] == pad_token_id

            return examples

        return data_collator

    def train(self) -> None:
        training_arguments = self.get_training_arguments()

        set_numpy_seed(training_arguments.seed)
        set_torch_seed(training_arguments.seed)

        with training_arguments.main_process_first():
            tokenizer = self.load_default_tokenizer()
            dataset_dict = self.load_dataset_dict(tokenizer)
            self._validate_dataset_dict(dataset_dict)
            model = self.get_model()
            self._call_init_weights(model, training_arguments.seed)

        data_collator = self.get_data_collator(tokenizer)
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
        expected_splits = ["train", "validation"]
        actual_splits = list(dataset_dict)
        assert set(expected_splits) == set(actual_splits), f"Expected dataset dict to have splits {expected_splits}, but got {actual_splits}."

        expected_columns = ["input_ids", "labels"]
        actual_columns = dataset_dict["train"].column_names
        assert set(expected_columns) == set(actual_columns), f"Expected dataset to have columns {expected_columns}, but got {actual_columns}."

    def _call_init_weights(self, model: TransformerBase, seed: int) -> None:
        set_torch_seed(seed)

        def init_weight_helper(module: Module):
            if not hasattr(module, "init_weights"): return
            try:
                module.init_weights()
            except NotImplementedError:
                pass

        model.apply(init_weight_helper)
