from typing import Any, Dict, Optional, Union

from abc import ABC, abstractmethod

from os.path import join

import json

from datasets import DatasetDict

from numpy.random import seed as set_numpy_seed

from torch import manual_seed as set_torch_seed, triu, ones
from torch.cuda import device_count
from torch.nn import Module, Parameter

from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

from transformers.utils.import_utils import is_torch_bf16_gpu_available
from transformers.integrations import TensorBoardCallback, rewrite_logs
from transformers.trainer_utils import EvalPrediction
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast, PrinterCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments

from march import CONFIG_DIR, RESULTS_DIR
from march.utils import log_run
from march.datasets.c4 import load_c4_tokenizer
from march.models.baseline import TransformerBase
from march.models.utils import count_parameters


class TensorboardWithCustomLogsCallback(TensorBoardCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero: return

        if self.tb_writer is None: self._init_summary_writer(args)

        if self.tb_writer is None: return  # Still None after _init_summary_writer

        logs = rewrite_logs(logs)

        for k, v in logs.items():
            if isinstance(v, (int, float)):
                self.tb_writer.add_scalar(k, v, state.global_step)
            elif isinstance(v, Parameter):
                self.tb_writer.add_histogram(k, v.data, state.global_step)
            else:  # For more complex values
                value_str = json.dumps(v)
                self.tb_writer.add_text(k, value_str, state.global_step)

        self.tb_writer.flush()


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.remove_callback(PrinterCallback)
        self.remove_callback(TensorBoardCallback)
        self.add_callback(TensorboardWithCustomLogsCallback)

    def _get_train_sampler(self) -> Optional[Sampler]:
        sampler = super()._get_train_sampler()
        assert type(sampler) is DistributedSampler

        return DistributedSampler(
            self.train_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            shuffle=False,
        )


class ExperimentBase(ABC):
    NUM_STEPS: Union[None, int] = None

    def __init__(self, batch_size_pow_scale: int=0, resume_from_checkpoint: bool=False) -> None:
        super().__init__()

        # For self.load_default_training_arguments
        self.batch_size_pow_scale = batch_size_pow_scale

        self.resume_from_checkpoint = resume_from_checkpoint

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
        with open(join(CONFIG_DIR, "default_training_arguments.json")) as default_training_args_file:
            args_dict = json.load(default_training_args_file)

        num_gpus = device_count()
        if num_gpus == 8:
            pass
        elif num_gpus == 4:
            args_dict["gradient_accumulation_steps"] = args_dict["gradient_accumulation_steps"] * 2
        else:
            raise ValueError(f"The number of GPUs must be 4 or 8, but got {num_gpus}")

        with open(join(CONFIG_DIR, "deepspeed.json")) as deepspeed_config_file:
            deepspeed_config = json.load(deepspeed_config_file)
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

        # Default to using bf16 if available
        use_bf16 = is_torch_bf16_gpu_available()
        args_dict["bf16"] = use_bf16
        args_dict["bf16_full_eval"] = use_bf16

        if self.NUM_STEPS is not None:
            assert isinstance(self.NUM_STEPS, int)
            args_dict["max_steps"] = self.NUM_STEPS

        # Output and logging directories
        args_dict["output_dir"] = self.output_dir
        args_dict["logging_dir"] = self.output_dir

        return args_dict

    def load_default_tokenizer(self) -> PreTrainedTokenizerFast:
        return load_c4_tokenizer()

    def get_data_collator(self, tokenizer: PreTrainedTokenizerFast):
        base_data_collator = DataCollatorForSeq2Seq(tokenizer)
        bos_token_id = tokenizer.bos_token_id
        pad_token_id = tokenizer.pad_token_id
        def data_collator(examples):
            for example in examples:
                example["decoder_input_ids"] = [bos_token_id] + example["labels"][:-1]

            examples = base_data_collator(examples)

            # Attention masks have values of 1 for should mask, 0 otherwise
            examples["attention_mask"] = examples["input_ids"] == pad_token_id

            batch_size, decoder_input_length = examples["decoder_input_ids"].size()
            causal_mask = triu(ones(decoder_input_length, decoder_input_length, dtype=bool), diagonal=1)

            decoder_attention_mask = examples["decoder_input_ids"] == pad_token_id
            decoder_attention_mask[:, 0] = 0  # in T5, bos_token_id == pad_token_id
            decoder_attention_mask = decoder_attention_mask[:, None, :] | causal_mask[None, :, :]
            assert decoder_attention_mask.size() == (batch_size, decoder_input_length, decoder_input_length), f"Expected decoder attention mask of shape {(batch_size, decoder_input_length, decoder_input_length)}, but got {decoder_attention_mask.size()}."

            examples["decoder_attention_mask"] = decoder_attention_mask

            return examples

        return data_collator

    def get_compute_metrics(self, model: TransformerBase):
        def compute_metrics(eval_preds: EvalPrediction) -> Dict[str, Any]:
            return dict(model.named_parameters())

        return compute_metrics

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
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_arguments,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.get_compute_metrics(model),
        )
        trainer.log({"num_params": count_parameters(model)})

        log_run(trainer.is_world_process_zero(), self.name)        

        trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)

        trainer.save_model()

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
