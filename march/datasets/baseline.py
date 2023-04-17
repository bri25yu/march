from typing import Dict, List

from datasets import DatasetDict, load_dataset

from tokenizers import Encoding, Tokenizer

from march.tokenization import EXTRA_ID_TOKENS, EOS_TOKEN
from march.datasets.span_corrupt_utils import create_span_corrupt_inputs


MASK_PROB = 0.15
AVERAGE_SPAN_LENGTH = 3


def load_wikitext103(tokenizer: Tokenizer) -> DatasetDict:
    sentinel_start_id = tokenizer.token_to_id(EXTRA_ID_TOKENS[-1])

    def tokenize_fn(examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
        outputs = {"input_ids": [], "labels": []}
        for text in examples["text"]:
            tokenized_output: Encoding = tokenizer.encode(text)
            input_ids, label_ids = create_span_corrupt_inputs(
                tokenized_output.ids, MASK_PROB, AVERAGE_SPAN_LENGTH, sentinel_start_id
            )
            outputs["input_ids"].append(input_ids)
            outputs["labels"].append(label_ids)

        return outputs

    dataset_dict = load_dataset("wikitext", "wikitext-103-raw-v1")
    return dataset_dict.map(
        tokenize_fn, batched=True, num_proc=16, remove_columns=dataset_dict["train"].column_names, desc="Tokenizing"
    )
