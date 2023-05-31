from os.path import join

from itertools import chain

import march  # Redirect cache

from datasets import DatasetDict, load_dataset, load_from_disk

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from march import CACHE_DIR
from march.datasets.span_corrupt_utils import DataCollatorForT5MLM, compute_input_and_target_lengths


# T5 span corruption parameters
MASK_PROB = 0.15
AVERAGE_SPAN_LENGTH = 3

MAX_LENGTH = 256

VOCAB_SIZE = 32128  # len(load_c4_tokenizer()) = 32000 vocab + 100 sentinel tokens rounded to the nearest multiple of 64

MULTIPROCESSING_NUM_PROC = 32


def load_c4_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=MAX_LENGTH, use_fast=False)
    tokenizer.bos_token_id = tokenizer.pad_token_id
    return tokenizer


def load_c4() -> DatasetDict:
    return load_dataset(f"hlillemark/c4_t5_corrupted_seqlen{MAX_LENGTH}")


def load_c4_text() -> DatasetDict:
    return load_dataset("c4", "en")


def load_c4_text_tiny() -> DatasetDict:
    num_train = 100_000
    num_validation = 10_000
    return DatasetDict({
        "train": load_dataset("c4", "en", split=f"train[:{num_train}]"),
        "validation": load_dataset("c4", "en", split=f"validation[:{num_validation}]"),
    })


def tokenize_using_t5(text_dataset_dict: DatasetDict, tokenized_dataset_path: str) -> DatasetDict:
    local_path = join(CACHE_DIR, tokenized_dataset_path)
    try:
        return load_from_disk(local_path)
    except FileNotFoundError:
        pass

    assert set(text_dataset_dict.keys()) == set(["train", "validation"])

    tokenizer = load_c4_tokenizer()
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


def pack_for_t5_span_corruption(
    tokenized_dataset_dict: DatasetDict,
    packed_dataset_path: str,
    max_seq_length: int,
    noise_density: float,
    mean_noise_span_length: float,
) -> DatasetDict:
    local_path = join(CACHE_DIR, packed_dataset_path)
    try:
        return load_from_disk(local_path)
    except FileNotFoundError:
        pass

    expanded_inputs_length, targets_length = compute_input_and_target_lengths(
        inputs_length=max_seq_length,
        noise_density=noise_density,
        mean_noise_span_length=mean_noise_span_length,
    )

    def pack_fn(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= expanded_inputs_length:
            total_length = (total_length // expanded_inputs_length) * expanded_inputs_length

        result = {
            k: [t[i : i + expanded_inputs_length] for i in range(0, total_length, expanded_inputs_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    packed_dataset_dict = tokenized_dataset_dict.map(
        pack_fn,
        batched=True,
        num_proc=MULTIPROCESSING_NUM_PROC,
    )

    packed_dataset_dict.save_to_disk(local_path)

    return packed_dataset_dict


def span_corrupt_packed_dataset(
    packed_dataset_dict: DatasetDict,
    corrupted_dataset_path: str,
    max_seq_length: int,
    noise_density: float,
    mean_noise_span_length: float,
) -> DatasetDict:
    local_path = join(CACHE_DIR, corrupted_dataset_path)
    try:
        return load_from_disk(local_path)
    except FileNotFoundError:
        pass

    tokenizer = load_c4_tokenizer()

    expanded_inputs_length, targets_length = compute_input_and_target_lengths(
        inputs_length=max_seq_length,
        noise_density=noise_density,
        mean_noise_span_length=mean_noise_span_length,
    )

    data_collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=noise_density,
        mean_noise_span_length=mean_noise_span_length,
        input_length=max_seq_length,
        target_length=targets_length,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
    )

    span_corrupted_dataset_dict = packed_dataset_dict.map(
        data_collator,
        batched=True,
        num_proc=MULTIPROCESSING_NUM_PROC,
    )

    span_corrupted_dataset_dict.save_to_disk(local_path)
    span_corrupted_dataset_dict.push_to_hub(corrupted_dataset_path)

    return span_corrupted_dataset_dict


def create_span_corrupted_c4(use_tiny: bool=False) -> DatasetDict:
    max_seq_length = MAX_LENGTH
    noise_density = MASK_PROB
    mean_noise_span_length = AVERAGE_SPAN_LENGTH

    tokenized_dataset_path = f"c4_t5_tokenized_seqlen{max_seq_length}"
    packed_dataset_path = f"c4_t5_packed_seqlen{max_seq_length}"
    corrupted_dataset_path = f"c4_t5_corrupted_seqlen{max_seq_length}"

    if use_tiny:
        text_dataset_dict = load_c4_text_tiny()

        dataset_path_suffix = "_tiny"
        tokenized_dataset_path += dataset_path_suffix
        packed_dataset_path += dataset_path_suffix
        corrupted_dataset_path += dataset_path_suffix
    else:
        text_dataset_dict = load_c4_text()

    tokenized_dataset_dict = tokenize_using_t5(
        text_dataset_dict=text_dataset_dict,
        tokenized_dataset_path=tokenized_dataset_path,
    )
    packed_dataset_dict = pack_for_t5_span_corruption(
        tokenized_dataset_dict=tokenized_dataset_dict,
        packed_dataset_path=packed_dataset_path,
        max_seq_length=max_seq_length,
        noise_density=noise_density,
        mean_noise_span_length=mean_noise_span_length,
    )
    corrupted_dataset_dict = span_corrupt_packed_dataset(
        packed_dataset_dict=packed_dataset_dict,
        corrupted_dataset_path=corrupted_dataset_path,
        max_seq_length=max_seq_length,
        noise_density=noise_density,
        mean_noise_span_length=mean_noise_span_length,
    )

    return corrupted_dataset_dict
