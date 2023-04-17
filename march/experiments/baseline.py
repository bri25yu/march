from datasets import DatasetDict

from tokenizers import Tokenizer

from transformers import Seq2SeqTrainingArguments

from march.datasets.baseline import load_wikitext103
from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig
from march.experiments.base import ExperimentBase


class BaselineExperiment(ExperimentBase):
    def load_dataset_dict(self, tokenizer: Tokenizer) -> DatasetDict:
        return load_wikitext103(tokenizer)

    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig(1024, 12, 64)
        return BaselineTransformer(config)
