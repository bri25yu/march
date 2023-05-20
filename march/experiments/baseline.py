from types import MethodType

from datasets import DatasetDict

from torch import manual_seed as set_torch_seed

from transformers import PreTrainedTokenizerFast, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoConfig

from march.datasets.c4 import load_c4, load_c4_full
from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig
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


class BaselineSmallFullTrain(BaselineExperiment):
    # This trains the model for 100k steps at 1m tokens per step = 100B tokens seen total
    NUM_STEPS = 100_000

    def load_dataset_dict(self, tokenizer: PreTrainedTokenizerFast) -> DatasetDict:
        return load_c4_full()

    def get_model(self) -> TransformerBase:
        config = TransformerConfig(
            dim_model=512,
            num_layers=12,
        )
        return BaselineTransformer(config)


class BaselineT5Small(BaselineT5Experiment):
    MODEL_NAME = "t5-small"
    # We don't technically need to use c4 full here, but we do so to match with t5 small full train
    def load_dataset_dict(self, tokenizer: PreTrainedTokenizerFast) -> DatasetDict:
        return load_c4_full()
