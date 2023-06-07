from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast

from march.datasets.c4 import load_c4_tokenizer
from march.models.baseline_v2 import BaselineV2Config, BaselineV2Transformer
from march.experiments.baseline import BaselineExperiment, TransformerBase


# BaselineV2 transformer doesn't take in attention masks
class BaselineV2Experiment(BaselineExperiment):
    def load_tokenizer(self) -> PreTrainedTokenizerFast:
        tokenizer = load_c4_tokenizer()
        tokenizer.model_input_names = ["input_ids"]
        return tokenizer

    def get_data_collator(self, tokenizer: PreTrainedTokenizerFast):
        base_data_collator = DataCollatorForSeq2Seq(tokenizer)
        bos_token_id = tokenizer.bos_token_id

        def data_collator(examples):
            for example in examples:
                example["decoder_input_ids"] = [bos_token_id] + example["labels"][:-1]

            return base_data_collator(examples)

        return data_collator

    def get_model(self) -> TransformerBase:
        config = BaselineV2Config()
        return BaselineV2Transformer(config)
