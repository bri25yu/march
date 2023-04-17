from typing import Dict, List

from datasets import DatasetDict, load_dataset

from transformers import PreTrainedTokenizerFast

from march.tokenization import EXTRA_ID_TOKENS, MAX_LENGTH
from march.datasets.span_corrupt_utils import create_span_corrupt_inputs


MASK_PROB = 0.15
AVERAGE_SPAN_LENGTH = 3

WIKITEXT103_BASELINE_NAME = "wikitext103_baseline"


def create_wikitext103_baseline(tokenizer: PreTrainedTokenizerFast) -> None:
    sentinel_start_id = tokenizer.convert_tokens_to_ids(EXTRA_ID_TOKENS[-1])

    delimiters = [" = ", " = = ", " = = = "]
    def filter_fn(example: Dict[str, str]) -> bool:
        text = example["text"]
        return len(text) > 0 and not any(text.startswith(delimiter) for delimiter in delimiters)

    def tokenize_fn(examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
        return {"input_ids": tokenizer(examples["text"]).input_ids}

    def pack_fn(examples: Dict[str, List[int]]) -> Dict[str, List[int]]:
        outputs = {"input_ids": []}
        current = []
        for input_ids in examples["input_ids"]:
            current.extend(input_ids)
            if len(current) > MAX_LENGTH:
                outputs["input_ids"].append(current[:MAX_LENGTH])
                current = current[MAX_LENGTH:]
        return outputs

    def apply_span_corruption(examples: Dict[str, List[int]]) -> Dict[str, List[int]]:
        outputs = {"input_ids": [], "labels": []}
        for input_ids in examples["input_ids"]:
            corrupted_input_ids, label_ids = create_span_corrupt_inputs(
                input_ids, MASK_PROB, AVERAGE_SPAN_LENGTH, sentinel_start_id
            )
            outputs["input_ids"].append(corrupted_input_ids)
            outputs["labels"].append(label_ids)

        return outputs


    dataset_dict = load_dataset("wikitext", "wikitext-103-raw-v1")
    examples = "\n\t".join(dataset_dict["train"]["text"][:5])
    print(f"Raw Wikitext-103\n{dataset_dict}\n\t{examples}")

    filtered_dataset_dict = dataset_dict.filter(filter_fn, num_proc=16)
    examples = "\n\t".join(filtered_dataset_dict["train"]["text"][:5])
    print(f"Filtered Wikitext-103\n{filtered_dataset_dict}\n\t{examples}")

    tokenized_dataset_dict = filtered_dataset_dict.map(tokenize_fn, batched=True, remove_columns=dataset_dict["train"].column_names, desc="Tokenizing", num_proc=16)
    print(f"Tokenized Wikitext-103\n{tokenized_dataset_dict}")

    packed_dataset_dict = tokenized_dataset_dict.map(pack_fn, batched=True, desc="Packing")
    print(f"Packed Wikitext-103\n{packed_dataset_dict}")

    span_corrupted_dataset_dict = packed_dataset_dict.map(apply_span_corruption, batched=True, num_proc=16, desc="Applying span corruption")
    print(f"Span corrupted Wikitext-103\n{span_corrupted_dataset_dict}")

    span_corrupted_dataset_dict.push_to_hub(WIKITEXT103_BASELINE_NAME)


def load_wikitext103_baseline() -> DatasetDict:
    return load_dataset("bri25yu/wikitext103_baseline")
