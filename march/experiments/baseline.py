from types import MethodType

from os.path import join

import json

from datasets import DatasetDict

from torch import manual_seed as set_torch_seed, triu, ones, cat, zeros

from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoConfig

from march import CONFIG_DIR
from march.datasets.c4 import load_c4
from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig
from march.models.DObaseline import DOBaselineTransformer
from march.experiments.base import ExperimentBase

class BaselineExperiment(ExperimentBase):
    def load_dataset_dict(self, tokenizer: PreTrainedTokenizerFast) -> DatasetDict:
        return load_c4()

    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        return Seq2SeqTrainingArguments(**default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return BaselineTransformer(config)


class BaselineT5Experiment(BaselineExperiment):
    MODEL_NAME = "t5-base"

    def get_model(self) -> TransformerBase:
        config = AutoConfig.from_pretrained(self.MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_config(config)
        setattr(model, "count_parameters", MethodType(BaselineTransformer.count_parameters, model))

        return model

    def get_data_collator(self, tokenizer: PreTrainedTokenizerFast):
        data_collator = super().get_data_collator(tokenizer)
        # Invert the mask for T5, change the pad token id
        def inverted_attention_mask_data_collator(examples):
            examples = data_collator(examples)
            examples["attention_mask"] = ~examples["attention_mask"]
            examples["decoder_attention_mask"] = ~examples["decoder_attention_mask"]

            return examples

        return inverted_attention_mask_data_collator

    def _call_init_weights(self, model: TransformerBase, seed: int) -> None:
        # Special HF T5 model weight re-init
        set_torch_seed(seed)

        model.apply(model._init_weights)


class BaselineT5LargeExperiment(BaselineT5Experiment):
    MODEL_NAME = "t5-large"


class BaselineLargeExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(
            dim_model=1024,
            num_layers=48,
        )
        return BaselineTransformer(config)


class BaselineSmallFullTrainExperiment(BaselineExperiment):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.resume_from_checkpoint = True

    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        args_dict = self.load_default_training_arguments()
        with open(join(CONFIG_DIR, "full_train_training_arguments.json")) as full_train_training_args_file:
            full_train_args_dict = json.load(full_train_training_args_file)

        args_dict.update(full_train_args_dict)

        return Seq2SeqTrainingArguments(**args_dict)

    def load_dataset_dict(self, tokenizer: PreTrainedTokenizerFast) -> DatasetDict:
        raise NotImplementedError

    def get_model(self) -> TransformerBase:
        config = TransformerConfig(
            dim_model=512,
            num_layers=12,
        )
        return BaselineTransformer(config)


class BaselineT5SmallExperiment(BaselineT5Experiment):
    MODEL_NAME = "t5-small"
    # We don't technically need to use c4 full here, but we do so to match with t5 small full train
    def load_dataset_dict(self, tokenizer: PreTrainedTokenizerFast) -> DatasetDict:
        raise NotImplementedError


class DOBaselineExperiment(BaselineExperiment):
    # We need to input the input_ids properly here with the DO model

    def get_data_collator(self, tokenizer: PreTrainedTokenizerFast):
        base_data_collator = DataCollatorForSeq2Seq(tokenizer)
        bos_token_id = tokenizer.bos_token_id
        pad_token_id = tokenizer.pad_token_id
        def data_collator(examples):
            # Do normal processing first: 
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

            # Then combine the attention mask and decoder attention mask into one by concatenating them
            # First we have to create the causal mask (0's) for the input attention mask
            _, input_length = examples["input_ids"].size()
            causal_mask_input = zeros(decoder_input_length, input_length, dtype=bool)
            attention_mask = examples['attention_mask'][:, None, :] | causal_mask_input[None, :, :]
            # We end up with a mask of shape (batch_size, decoder_input_length, input_length + decoder_input_length)
            # This makes sense since we have decoder_input_length steps of decoding, but have to provide this 
            # attention mask over all possible tokens, including the input tokens, which have length input_length
            examples["attention_mask"] = cat([attention_mask, decoder_attention_mask], dim=-1)

            # and the decoder input ids and input ids into one by concatenating them
            examples["input_ids"] = cat([examples["input_ids"], examples["decoder_input_ids"]], dim=-1)

            return examples

        return data_collator

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return DOBaselineTransformer(config)
