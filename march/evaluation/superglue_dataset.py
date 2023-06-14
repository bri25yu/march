from typing import Dict, List, Union

from os.path import join

from dataclasses import dataclass

import json

from math import ceil

from datasets import DatasetDict, concatenate_datasets, load_dataset

from transformers import PreTrainedTokenizer, set_seed

from march import CONFIG_DIR
from march.datasets.c4 import load_c4_tokenizer


num_proc = 4
max_steps = 10_000
examples_per_step = 64
total_examples = max_steps * examples_per_step  # 64 * 10_000 = 640_000


@dataclass
class SuperGLUETaskProcessor:
    task_name: str
    convert_to_label: Union[Dict[int, str], str]
    metric_for_best_model: str
    prompt: str

    def __post_init__(self) -> None:
        assert not isinstance(self.convert_to_label, (dict, str)), f"Convert to label must be a dict mapping or str to format, but got {type(self.convert_to_label)}"

    def convert_inputs_to_label(self, inputs: Dict[str, str]) -> str:
        if isinstance(self.convert_to_label, dict):
            return self.convert_to_label[str(inputs["label"])]  # We use str(label) because the dictionary was stored in json using a string key
        elif isinstance(self.convert_to_label, str):
            return self.convert_to_label.format(**inputs)


def transpose(inputs: Dict[str, List[str]]) -> List[Dict[str, str]]:
    keys = list(inputs.keys())
    n = len(inputs[keys[0]])
    return [{k: inputs[k][i] for k in keys} for i in range(n)]


def process_single_task(task_processor: SuperGLUETaskProcessor, tokenizer: PreTrainedTokenizer, num_examples: int) -> DatasetDict:
    dataset_dict = load_dataset("super_glue", task_processor.task_name)
    dataset_dict.pop("test")

    def map_fn(inputs: Dict[str, List[str]]) -> Dict[str, List[int]]:
        inputs = transpose(inputs)
        text_input = [task_processor.prompt.format(**d) for d in inputs]
        text_label = [task_processor.convert_inputs_to_label(d) for d in inputs]
        return {
            "input_ids": tokenizer(text_input, return_attention_mask=False).input_ids,
            "labels": tokenizer(text_label, return_attention_mask=False).input_ids,
        }

    tokenized_dataset_dict = dataset_dict.map(
        map_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset_dict["train"].column_names,
        desc=f"Converting and tokenizing inputs for {task_processor.task_name}",
    )

    num_train_repeats = ceil(num_examples / len(tokenized_dataset_dict["train"]))
    if num_train_repeats > 1:
        tokenized_dataset_dict["train"] = concatenate_datasets([tokenized_dataset_dict["train"]] * num_train_repeats)

    return tokenized_dataset_dict


def process_superglue(tokenizer: PreTrainedTokenizer, dataset_path: str, seed: int=42):
    with open(join(CONFIG_DIR, "superglue_task_processors.json")) as f:
        task_processors = json.load(f)
    task_processors = [SuperGLUETaskProcessor(**d) for d in task_processors]

    ex_per_task = total_examples // task_processors
    dataset_dicts = {t.task_name: process_single_task(t, tokenizer, ex_per_task) for t in task_processors}
    set_seed(seed)  # Shuffle examples by task
    dataset_dict = DatasetDict({
        "train": concatenate_datasets([d["train"] for d in dataset_dicts]).shuffle().flatten_indices(),
        **{f"validation_{name}": d["validation"] for name, d in dataset_dicts.items()},
    })
    dataset_dict.push_to_hub(dataset_path)


def process_superglue_t5() -> None:
    tokenizer = load_c4_tokenizer()
    dataset_path = "superglue_t5like"
    process_superglue(tokenizer, dataset_path)


def load_superglue() -> DatasetDict:
    return load_dataset("bri25yu/superglue_t5like")
