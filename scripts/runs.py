from typing import Dict, Iterable, Optional, Type

from os.path import exists, join

from collections import OrderedDict

from time import time
from datetime import timedelta

from tqdm.auto import tqdm

from pandas import DataFrame

from tensorboard.backend.event_processing.directory_watcher import DirectoryDeletedError
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
    ScalarEvent,
)

from march import ROOT_DIR
from march.experiments import ExperimentBase, available_experiments


RESULTS_DIRS = [
    join(ROOT_DIR, "..", "archived_results"),
    join(ROOT_DIR, "..", "results"),
]


class ExperimentResult(OrderedDict):
    def __init__(
        self,
        experiment: ExperimentBase,
        hostname: str,
        total_steps: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.experiment = experiment
        self.name = experiment.name.removesuffix("Experiment")
        self.hostname = hostname
        self.total_steps = total_steps

    @classmethod
    def from_experiment(
        cls, experiment: ExperimentBase
    ) -> "Optional[ExperimentResult]":
        results_path = None
        for results_dir in RESULTS_DIRS:
            results_path = join(results_dir, experiment.name)
            if not exists(results_path):
                results_path = None
            else:
                break

        if results_path is None:
            return

        event_accumulator = EventAccumulator(results_path)
        try:
            event_accumulator.Reload()
        except DirectoryDeletedError:
            return

        try:
            scalars: Iterable[ScalarEvent] = event_accumulator.Scalars("train/loss")
        except KeyError:
            return

        try:
            hostname = event_accumulator.Tensors("hostname/text_summary")[
                0
            ].tensor_proto.string_val[0].decode()
        except KeyError:
            hostname = "Unknown"

        total_steps = experiment.load_default_training_arguments()["max_steps"]

        return cls(experiment, hostname, total_steps, [(s.step, s) for s in scalars])

    def is_finished(self) -> bool:
        last_step = max(self)
        return last_step == self.total_steps

    def is_old(self) -> bool:
        last_scalarevent: ScalarEvent = self[max(self)]
        last_timestamp = last_scalarevent.wall_time  # In seconds
        diff_min = (time() - last_timestamp) / 60
        return diff_min >= 30

    def is_ongoing(self) -> bool:
        return not self.is_finished() and not self.is_old()

    def time_left(self) -> str:
        if self.is_finished():
            return "Finished"

        first_scalarevent = self[min(self)]
        last_scalarevent = self[max(self)]
        steps_run = last_scalarevent.step - first_scalarevent.step
        seconds_elapsed = last_scalarevent.wall_time - first_scalarevent.wall_time
        seconds_per_step = seconds_elapsed / steps_run
        steps_left = self.total_steps - steps_run
        seconds_left = steps_left * seconds_per_step

        diff = timedelta(seconds=seconds_left)
        hours, rem = divmod(diff.seconds, 3600)
        minutes, _ = divmod(rem, 60)

        return (f"{diff.days}d " if diff.days else "") + f"{hours:02}hr {minutes:02}min"

    def get_summary(self) -> Dict[str, str]:
        get_loss = lambda step: f"{self[step].value:.3f}" if step in self else ""
        return {
            "Experiment name": self.name,
            "Node": self.hostname,
            "Step": str(max(self)),
            "Time left": self.time_left(),
            "Loss at 1k": get_loss(1_000),
            "Loss at 5k": get_loss(5_000),
            "Loss at 10k": get_loss(10_000),
            "Loss at 15k": get_loss(15_000),
        }


class BatchExperimentResults(dict):
    @classmethod
    def from_available_experiments(cls) -> "BatchExperimentResults":
        exp_cls: Type[ExperimentBase]
        available_exp_results = cls()
        for exp_cls in tqdm(available_experiments.values(), desc="Reading results"):
            exp_result: ExperimentResult = ExperimentResult.from_experiment(exp_cls())
            if exp_result is not None:
                available_exp_results[exp_result.name] = exp_result

        return available_exp_results

    def get_summary(self) -> DataFrame:
        df = DataFrame.from_dict(list(map(lambda v: v.get_summary(), self.values())))
        df = df.set_index("Experiment name")

        df_str = df.to_string() + "\n"
        with open(join(ROOT_DIR, "..", "runs.txt"), "w") as f:
            f.write(df_str)
        print(df_str)

        return df


exp_results = BatchExperimentResults.from_available_experiments()
exp_results.get_summary()
