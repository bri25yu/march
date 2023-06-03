from os.path import join

from itertools import chain

from datasets import DatasetDict, load_dataset, load_from_disk

from transformers.tokenization_utils_base import VERY_LARGE_INTEGER
from transformers import LlamaTokenizer

from march import CACHE_DIR
from march.datasets.c4 import MAX_LENGTH, MULTIPROCESSING_NUM_PROC, load_c4_text, load_c4_text_tiny


LLAMA_VOCAB_SIZE = 32000  # len(load_llama_tokenizer()) = 32000


def load_llama_tokenizer() -> LlamaTokenizer:
    tokenizer = LlamaTokenizer.from_pretrained(
        "decapoda-research/llama-7b-hf", model_max_length=MAX_LENGTH, use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.bos_token_id
    return tokenizer


def load_c4_llama(num_train_examples: int=None, num_validation_examples: int=None) -> DatasetDict:
    train_split = f"[:{num_train_examples}]" if num_train_examples is not None else ""
    validation_split = f"[:{num_validation_examples}]" if num_validation_examples is not None else ""

    return DatasetDict({
        "train": load_dataset(f"hlillemark/c4_llama_packed_seqlen{MAX_LENGTH}", split=f"train{train_split}"),
        "validation": load_dataset(f"hlillemark/c4_llama_packed_seqlen{MAX_LENGTH}", split=f"validation{validation_split}"),
    })


def tokenize_using_llama(
    text_dataset_dict: DatasetDict, tokenized_dataset_path: str
) -> DatasetDict:
    local_path = join(CACHE_DIR, tokenized_dataset_path)
    try:
        return load_from_disk(local_path)
    except FileNotFoundError:
        pass

    assert set(text_dataset_dict.keys()) == set(["train", "validation"])

    tokenizer = load_llama_tokenizer()

    # Just for tokenization to avoid seqlen messages
    # Our max seq len is not caused by the model, it's a custom constraint, so no need to worry about it during tokenization
    tokenizer.model_max_length = VERY_LARGE_INTEGER

    def tokenize_fn(examples):
        return tokenizer(examples["text"], return_attention_mask=False)

    tokenized_dataset_dict = text_dataset_dict.map(
        tokenize_fn,
        batched=True,
        num_proc=MULTIPROCESSING_NUM_PROC,
        remove_columns=text_dataset_dict["train"].column_names,
    )
    tokenized_dataset_dict.save_to_disk(local_path)

    return tokenized_dataset_dict


def pack_for_language_generation(
    tokenized_dataset_dict: DatasetDict,
    packed_dataset_path: str,
    max_seq_length: int,
) -> DatasetDict:
    local_path = join(CACHE_DIR, packed_dataset_path)
    try:
        return load_from_disk(local_path)
    except FileNotFoundError:
        pass

    def pack_fn(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= max_seq_length:
            total_length = (
                total_length // max_seq_length
            ) * max_seq_length

        result = {
            k: [
                t[i : i + max_seq_length]
                for i in range(0, total_length, max_seq_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    packed_dataset_dict = tokenized_dataset_dict.map(
        pack_fn,
        batched=True,
        num_proc=MULTIPROCESSING_NUM_PROC,
    )

    packed_dataset_dict.save_to_disk(local_path)
    packed_dataset_dict.push_to_hub(packed_dataset_path)

    return packed_dataset_dict


def create_c4_for_language_generation(use_tiny: bool = False) -> DatasetDict:
    max_seq_length = MAX_LENGTH

    tokenized_dataset_path = f"c4_llama_tokenized_seqlen{max_seq_length}"
    packed_dataset_path = f"c4_llama_packed_seqlen{max_seq_length}"

    if use_tiny:
        text_dataset_dict = load_c4_text_tiny()

        dataset_path_suffix = "_tiny"
        tokenized_dataset_path += dataset_path_suffix
        packed_dataset_path += dataset_path_suffix
    else:
        text_dataset_dict = load_c4_text()

    tokenized_dataset_dict = tokenize_using_llama(
        text_dataset_dict=text_dataset_dict,
        tokenized_dataset_path=tokenized_dataset_path,
    )

    assert set(tokenized_dataset_dict.keys()) == set(["train", "validation"])
    assert set(tokenized_dataset_dict["train"].column_names) == set(["input_ids"])

    n_head = 10_000
    mean_length = (
        sum(
            map(len, tokenized_dataset_dict["train"].select(range(n_head))["input_ids"])
        )
        / n_head
    )
    print(
        f"After tokenization, average length of the first {n_head:,} train split examples is {mean_length}"
    )

    packed_dataset_dict = pack_for_language_generation(
        tokenized_dataset_dict=tokenized_dataset_dict,
        packed_dataset_path=packed_dataset_path,
        max_seq_length=max_seq_length,
    )

    assert set(packed_dataset_dict.keys()) == set(["train", "validation"])
    assert set(packed_dataset_dict["train"].column_names) == set(["input_ids"])

    mean_length = (
        sum(map(len, packed_dataset_dict["train"].select(range(n_head))["input_ids"]))
        / n_head
    )
    print(
        f"After packing, average length of the first {n_head:,} train split examples is {mean_length}"
    )
    print(f"This number should be {max_seq_length}")

    return packed_dataset_dict
