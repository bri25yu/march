from typing import Union

import os


__all__ = ["run"]


os.environ["TOKENIZERS_PARALLELISM"] = "false"


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, "..", "results")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
CACHE_DIR = os.path.join(ROOT_DIR, "..", "cache")


os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR


def run(
    experiment_name: Union[None, str]=None,
    batch_size_pow_scale: int=0,
    use_fp32: bool=False
) -> None:
    """
    batch_size_pow_scale: int
        Will multiply the default batch size by 2 ** batch_size_pow_scale.
        e.g. if original_batch_size = 8, batch_size_pow_scale = 2
            => new_batch_size = original_batch_size * (2 ** batch_size_pow_scale)
                = 8 * (2 ** 2) = 32
        Default value is 0 or a scaling factor of 2 ** 0 = 1.
    use_fp32: bool
        The default experiment precision is bf16. Setting this flag to true will turn off
        bf16 and use fp32 instead.

    """
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

    original_pow_scale = batch_size_pow_scale
    finished = False
    while not finished:
        try:
            experiment = experiment_cls(batch_size_pow_scale=batch_size_pow_scale, use_fp32=use_fp32)
            experiment.train()
            finished = True
        except RuntimeError as e:
            error_message = str(e)
            if "CUDA out of memory" in error_message:
                batch_size_pow_scale -= 1
                print(f"Lowering batch size by a factor of 2. Original pow scale {original_pow_scale}, current pow scale {batch_size_pow_scale}")
            else:
                raise e
        except AssertionError as e:
            raise e
