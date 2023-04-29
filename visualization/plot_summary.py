from typing import Dict, List, Tuple

from os import listdir
from os.path import abspath, dirname, exists, isdir, join

from tqdm.auto import tqdm

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib.pyplot as plt


__all__ = ["plot_comparative_experiment"]


ROOT_DIR = join(dirname(abspath(__file__)), "..")
RESULTS_DIRS = [join(ROOT_DIR, "archived_results"), join(ROOT_DIR, "results")]
VISUALIZATION_OUTPUT_DIR = join(ROOT_DIR, "readme_resources")


EXP_NAMES_TO_PATH: Dict[str, str] = dict()
for result_dir in RESULTS_DIRS:
    for name in listdir(result_dir):
        path = join(result_dir, name)
        if not isdir(path): continue

        EXP_NAMES_TO_PATH[name] = path


def get_property_values(path: str, property_name: str) -> Tuple[List[float], List[float]]:
    event_accumulator = EventAccumulator(path)
    event_accumulator.Reload()

    scalars = event_accumulator.Scalars(property_name)

    steps = [s.step for s in scalars]
    values = [s.value for s in scalars]

    return steps, values


def plot_comparative_experiment(
    experiment_names: List[str],
    legend_labels: List[str],
    title: str,
    save_name: str,
) -> None:
    property_names = ["train/loss", "eval/loss"]
    y_labels = ["Train loss", "Eval loss"]

    if not save_name.endswith(".png"): save_name += ".png"
    save_path = join(VISUALIZATION_OUTPUT_DIR, save_name)
    if exists(save_path):
        print(f"Already have a graph at {save_path}")
        return

    rows, cols = 1, len(property_names)
    fig, axs = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    for experiment_name, legend_label in tqdm(zip(experiment_names, legend_labels), desc="Plotting", total=len(experiment_names)):
        experiment_path = EXP_NAMES_TO_PATH[experiment_name]

        num_params = get_property_values(experiment_path, "train/num_params")[1][0]
        num_params_M = round(num_params / (1e6))  # Assume xxxM params
        legend_label_full = f"{legend_label} ({num_params_M}M params)"

        for property_name, ax in zip(property_names, axs):
            steps, values = get_property_values(experiment_path, property_name)
            ax.plot(steps, values, label=legend_label_full)

    for y_label, ax in zip(y_labels, axs):
        ax.set_ylim(ax.get_ylim()[0], 7.0)
        ax.set_xlabel("Steps")
        ax.set_ylabel(y_label)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path)
