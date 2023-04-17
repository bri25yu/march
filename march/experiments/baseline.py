from datasets import DatasetDict

from transformers import PreTrainedTokenizerFast, Seq2SeqTrainingArguments

from march.datasets.wikitext103 import load_wikitext103_baseline
from march.models.baseline import TransformerBase, BaselineTransformer, TransformerConfig
from march.experiments.base import ExperimentBase


class BaselineWikiTextExperiment(ExperimentBase):
    def load_dataset_dict(self, tokenizer: PreTrainedTokenizerFast) -> DatasetDict:
        return load_wikitext103_baseline()

    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return BaselineTransformer(config)
