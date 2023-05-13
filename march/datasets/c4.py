from typing import Dict, List

from datasets import DatasetDict, load_dataset

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from march.datasets.span_corrupt_utils import create_span_corrupt_inputs


MASK_PROB = 0.15
AVERAGE_SPAN_LENGTH = 3

# 10k steps * 1024 examples per batch = 10,240,000
NUM_TRAIN_EXAMPLES = 10_240_000
NUM_VAL_EXAMPLES = 10_000
C4_T5_10M_NAME = "c4_t5_10m"  # C4 with 10M train examples

EOS_TOKEN = "</s>"
EXTRA_ID_TOKENS = [f"<extra_id_{i}>" for i in reversed(range(100))]
MAX_LENGTH = 1024

VOCAB_SIZE = 32128  # len(load_c4_tokenizer()) = 32000 vocab + 100 sentinel tokens rounded to the nearest multiple of 64


def load_c4_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=1024)
    return tokenizer


def create_c4_10m(test: bool=False) -> None:
    tokenizer: PreTrainedTokenizerFast = load_c4_tokenizer()

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

    if test:
        dataset_dict = DatasetDict({
            "train": load_dataset("c4", "en", split="train[:100000]"),
            "validation": load_dataset("c4", "en", split="validation[:10000]"),
        })
    else:
        dataset_dict = DatasetDict({
            "train": load_dataset("c4", "en", split=f"train[:{3 * NUM_TRAIN_EXAMPLES}]"),
            "validation": load_dataset("c4", "en", split=f"validation[:{3 * NUM_VAL_EXAMPLES}]"),
        })
    print(f"Raw C4\n{dataset_dict}")

    tokenized_dataset_dict = dataset_dict.map(tokenize_fn, batched=True, remove_columns=dataset_dict["train"].column_names, desc="Tokenizing", num_proc=16)
    print(f"Tokenized C4\n{tokenized_dataset_dict}")

    packed_dataset_dict = tokenized_dataset_dict.map(pack_fn, batched=True, desc="Packing", num_proc=16)
    packed_dataset_dict = DatasetDict({
        "train": packed_dataset_dict["train"].select(range(NUM_TRAIN_EXAMPLES)),
        "validation": packed_dataset_dict["validation"].select(range(NUM_VAL_EXAMPLES)),
    })
    print(f"Packed C4\n{packed_dataset_dict}")

    span_corrupted_dataset_dict = packed_dataset_dict.map(apply_span_corruption, batched=True, num_proc=16, desc="Applying span corruption")
    print(f"Span corrupted C4\n{span_corrupted_dataset_dict}")

    if test:
        push_to_hub_name = f"{C4_T5_10M_NAME}_test"
    else:
        push_to_hub_name = C4_T5_10M_NAME
    span_corrupted_dataset_dict.push_to_hub(push_to_hub_name)


def load_c4() -> DatasetDict:
    return load_dataset(f"hlillemark/{C4_T5_10M_NAME}")
