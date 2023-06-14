from os import environ
from os.path import abspath, dirname, join


__all__ = ["run"]


environ["TOKENIZERS_PARALLELISM"] = "false"


ROOT_DIR = dirname(abspath(__file__))
RESULTS_DIR = join(ROOT_DIR, "..", "results")
EVALUATION_RESULTS_DIR = join(ROOT_DIR, "..", "evaluation_results")
CONFIG_DIR = join(ROOT_DIR, "config")
CACHE_DIR = join(ROOT_DIR, "..", "cache")


environ["TRANSFORMERS_CACHE"] = CACHE_DIR
environ["HF_DATASETS_CACHE"] = CACHE_DIR


def run(
    experiment_name: str,
    batch_size_pow_scale: int = 0,
    resume_from_checkpoint: bool = False,
    overwrite_old_experiment: bool = False,
) -> None:
    """
    batch_size_pow_scale: int
        Will multiply the default batch size by 2 ** batch_size_pow_scale.
        e.g. if original_batch_size = 8, batch_size_pow_scale = 2
            => new_batch_size = original_batch_size * (2 ** batch_size_pow_scale)
                = 8 * (2 ** 2) = 32
        Default value is 0 or a scaling factor of 2 ** 0 = 1.

    """
    from march.experiments import available_experiments

    if experiment_name not in available_experiments:
        raise ValueError(f"Experiment {experiment_name} is not recognized!")

    experiment_cls = available_experiments[experiment_name]
    original_pow_scale = batch_size_pow_scale
    finished = False
    while not finished:
        try:
            experiment = experiment_cls(
                batch_size_pow_scale=batch_size_pow_scale,
                resume_from_checkpoint=resume_from_checkpoint,
                overwrite_old_experiment=overwrite_old_experiment,
            )
            experiment.train()
            finished = True
        except RuntimeError as e:
            error_message = str(e)
            if "CUDA out of memory" in error_message:
                batch_size_pow_scale -= 1
                overwrite_old_experiment = True
                print(
                    f"Lowering batch size by a factor of 2. Original pow scale {original_pow_scale}, current pow scale {batch_size_pow_scale}"
                )
            else:
                raise e
