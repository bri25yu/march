from typing import Dict, List, Tuple, Optional

from os import listdir
from os.path import abspath, dirname, exists, isdir, join

from argparse import ArgumentParser

from tqdm.auto import tqdm

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from numpy import array, isin

from pandas import DataFrame

from matplotlib import rcParams

FONT_SIZE = 14
rcParams.update({"font.size": FONT_SIZE})
from matplotlib.ticker import FuncFormatter
from matplotlib.pyplot import subplot_mosaic
from matplotlib.font_manager import FontProperties

from march.visualization.compute_scaling_law import ScalingLawForCompute


__all__ = ["plot_comparative_experiment"]


ROOT_DIR = join(dirname(abspath(__file__)), "..", "..")
RESULTS_DIRS = [join(ROOT_DIR, "archived_results"), join(ROOT_DIR, "results")]
VISUALIZATION_OUTPUT_DIR = join(ROOT_DIR, "readme_resources")


EXP_NAMES_TO_PATH: Dict[str, str] = dict()
for result_dir in RESULTS_DIRS:
    for name in listdir(result_dir):
        path = join(result_dir, name)
        if not isdir(path):
            continue

        EXP_NAMES_TO_PATH[name] = path


def steps_log_scale_format_fn(tick_value, position):
    tick_value = int(tick_value)
    if tick_value < 1000:
        return f"{tick_value}"
    elif tick_value < 1_000_000:
        tick_value = tick_value // 1000
        return f"{tick_value}k"
    else:
        tick_value = tick_value // 1_000_000
        return f"{tick_value}M"


steps_log_scale_formatter = FuncFormatter(steps_log_scale_format_fn)


def get_property_values(
    path: str, property_name: str
) -> Tuple[List[float], List[float]]:
    event_accumulator = EventAccumulator(path)
    event_accumulator.Reload()

    scalars = event_accumulator.Scalars(property_name)

    steps = array([s.step for s in scalars])
    values = array([s.value for s in scalars])

    return steps, values


def plot_comparative_experiment(
    experiment_names: List[str],
    legend_labels: List[str],
    title: str,
    save_name: str,
    max_steps: Optional[int]=None,
) -> None:
    y_labels = [
        "Train loss",
        "Extrapolated train loss",
    ]
    ax_titles = [
        "Train loss curves over 15k steps",
        "Extrapolated train loss over 1M step budget",
    ]

    if not save_name.endswith(".pdf"):
        save_name += ".pdf"
    save_path = join(VISUALIZATION_OUTPUT_DIR, save_name)

    parser = ArgumentParser()
    parser.add_argument("--overwrite_existing", action="store_true")
    args = parser.parse_args()

    if not args.overwrite_existing and exists(save_path):
        print(f"Already have a graph at {save_path}")
        return

    rows, cols = 3, 4
    mosaic = [
        ["train_loss_ax", "scale_all_ax"],
        ["train_loss_ax", "scale_all_ax"],
        ["table_ax", "table_ax"],
    ]
    fig, axs_dict = subplot_mosaic(mosaic, figsize=(5 * cols, 4 * rows))
    train_loss_ax = axs_dict["train_loss_ax"]
    scaling_law_all_ax = axs_dict["scale_all_ax"]
    table_ax = axs_dict["table_ax"]
    axs = [train_loss_ax, scaling_law_all_ax]

    stats_df = DataFrame(
        columns=[
            "Experiment",
            "# params",
            "Train - eval loss",
            "Scaling law",
            "PPL at 100k",
            "PPL at 300k",
            "PPL at 1M",
        ]
    )

    for experiment_name, legend_label in tqdm(
        zip(experiment_names, legend_labels),
        desc="Plotting",
        total=len(experiment_names),
    ):
        experiment_path = EXP_NAMES_TO_PATH[experiment_name]

        train_steps, train_losses = get_property_values(experiment_path, "train/loss")
        in_range = train_steps <= max_steps
        train_steps = train_steps[in_range]
        train_losses = train_losses[in_range]
        train_loss_ax.plot(train_steps, train_losses, label=legend_label)

        scaling_law_plotter = ScalingLawForCompute(
            train_steps, train_losses, legend_label, stats_df
        )
        scaling_law_plotter.plot_over_all(scaling_law_all_ax)

        eval_steps, eval_losses = get_property_values(experiment_path, "eval/loss")
        train_steps_matched = isin(train_steps, eval_steps, assume_unique=True)
        eval_steps_matched = isin(eval_steps, train_steps, assume_unique=True)
        train_loss_matched = train_losses[train_steps_matched]
        eval_loss_matched = eval_losses[eval_steps_matched]
        diff_mean = (train_loss_matched - eval_loss_matched).mean()
        stats_df.at[len(stats_df.index) - 1, "Train - eval loss"] = f"{diff_mean:.3f}"

        num_params = get_property_values(experiment_path, "num_params")[1][0]
        num_params_M = f"{round(num_params / (1e6))}M"  # Assume xxxM params
        stats_df.at[len(stats_df.index) - 1, "# params"] = num_params_M

    for y_label, ax_title, ax in zip(y_labels, ax_titles, axs):
        ax.set_ylim(ax.get_ylim()[0], 7.0)
        ax.set_xlabel("Steps")
        ax.set_ylabel(y_label)
        ax.set_title(ax_title)
        ax.legend()

    scaling_law_all_ax.autoscale()
    scaling_law_all_ax.set_xscale("log")
    scaling_law_all_ax.set_xlabel("Steps with a log scale")
    scaling_law_all_ax.xaxis.set_major_formatter(steps_log_scale_formatter)

    x_margin = 0.0
    table = table_ax.table(
        cellText=stats_df.values,
        cellLoc="center",
        colLabels=stats_df.columns,
        bbox=[x_margin, 0, 1 - 2 * x_margin, 1],
    )
    table_ax.axis("off")
    table_ax.set_title("Scaling law fit details and perplexity (PPL) predictions")
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_SIZE + 4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontproperties=FontProperties(weight="bold"))

    table.auto_set_column_width(col=list(range(len(stats_df.index))))

    fig.suptitle(f"{title}\n", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, format="pdf")
