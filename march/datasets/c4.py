from typing import Dict, List

from datasets import DatasetDict, load_dataset

from transformers import PreTrainedTokenizerFast, AutoTokenizer

from march.tokenization import EXTRA_ID_TOKENS, MAX_LENGTH, load_c4_tokenizer
from march.datasets.span_corrupt_utils import create_span_corrupt_inputs


MASK_PROB = 0.15
AVERAGE_SPAN_LENGTH = 3

C4_T5_NAME = "c4_t5"


def create_c4(tokenizer: PreTrainedTokenizerFast) -> None:
    sentinel_start_id = tokenizer.convert_tokens_to_ids(EXTRA_ID_TOKENS[-1])

    def tokenize_fn(examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
        return {"input_ids": tokenizer(examples["text"]).input_ids}

    def pack_fn(examples: Dict[str, List[int]]) -> Dict[str, List[int]]:
        outputs = {"input_ids": []}
        current = []
        for input_ids in examples["input_ids"]:
            current.extend(input_ids)
            while len(current) > MAX_LENGTH:
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


    dataset_dict = load_dataset("c4", "en")
    print(f"Raw C4\n{dataset_dict}")

    tokenized_dataset_dict = dataset_dict.map(tokenize_fn, batched=True, remove_columns=dataset_dict["train"].column_names, desc="Tokenizing", num_proc=16)
    print(f"Tokenized C4\n{tokenized_dataset_dict}")

    packed_dataset_dict = tokenized_dataset_dict.map(pack_fn, batched=True, desc="Packing", num_proc=16)
    print(f"Packed C4\n{packed_dataset_dict}")

    span_corrupted_dataset_dict = packed_dataset_dict.map(apply_span_corruption, batched=True, num_proc=16, desc="Applying span corruption")
    print(f"Span corrupted C4\n{span_corrupted_dataset_dict}")

    span_corrupted_dataset_dict.push_to_hub(C4_T5_NAME)


def load_c4() -> DatasetDict:
    return load_dataset(f"hlillemark/{C4_T5_NAME}")
