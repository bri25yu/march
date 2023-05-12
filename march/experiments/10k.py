from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast, Seq2SeqTrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

from datasets import DatasetDict

from march.experiments.base import ExperimentBase
from march.tokenization import load_c4_tokenizer
from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig
from march.datasets.c4 import load_c4


class BaselineExperiment_10k(ExperimentBase):
    def load_dataset_dict(self, tokenizer: PreTrainedTokenizerFast) -> DatasetDict:
        return load_c4()

    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments["eval_steps"] = 200
        default_training_arguments["max_steps"] = 10000
        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def load_default_tokenizer(self) -> PreTrainedTokenizerFast:
        return load_c4_tokenizer()

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return BaselineTransformer(config)
