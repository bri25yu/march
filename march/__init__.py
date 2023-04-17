from typing import Union

import os


__all__ = ["run"]


os.environ["TOKENIZERS_PARALLELISM"] = "false"


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, "..", "results")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
DATASET_CACHE_DIR = os.path.join(ROOT_DIR, "..", "dataset_cache")
HUGGINGFACE_CACHE_DIR = os.path.join(ROOT_DIR, "..", "..", "huggingface_cache")


os.environ["TRANSFORMERS_CACHE"] = HUGGINGFACE_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = HUGGINGFACE_CACHE_DIR


def run(experiment_name: Union[None, str]=None) -> None:
    from march.experiments import available_experiments


    if experiment_name is None:
        experiment_names = list(available_experiments)
        experiment_index = input(
            "Choose an experiment class:\n" + "\n".join(f"{i+1}. {name}" for i, name in enumerate(experiment_names)) + "\n"
        )
        try:
            experiment_index = int(experiment_index) - 1
        except ValueError as e:
            raise e

        experiment_name = experiment_names[experiment_index]
        print(f"{experiment_name} chosen!")

    experiment_cls = available_experiments.get(experiment_name, None)

    if experiment_cls is None:
        print(f"Experiment {experiment_name} is not recognized")
        return

    experiment = experiment_cls()
    experiment.train()
