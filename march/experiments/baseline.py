from types import MethodType

from datasets import DatasetDict

from torch import manual_seed as set_torch_seed

from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast, Seq2SeqTrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

from march.datasets.c4 import load_c4
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


class BaselineFP32Experiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()

        default_training_arguments["bf16"] = False

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)


class BaselineT5Experiment(BaselineExperiment):
    MODEL_NAME = "t5-base"

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
