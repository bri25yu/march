from typing import List

from os.path import exists

from dataclasses import dataclass

from datetime import datetime

import pickle

from socket import gethostname

from march import RUN_LOG_PATH


__all__ = ["log_run", "print_most_recent_runs"]


@dataclass
class RunLog:
    experiment_name: str
    timestamp: datetime
    hostname: str

    timestamp_format_str: str = "%a %b %d %Y %I:%M%p"

    @property
    def timestamp_str(self) -> str:
        return self.timestamp.strftime(self.timestamp_format_str)

    @property
    def started_ago(self) -> str:
        diff = datetime.now() - self.timestamp
        hours, rem = divmod(diff.seconds, 3600)
        minutes, _ = divmod(rem, 60)

        return f"{diff.days} days" * diff.days + f"{hours} hr {minutes} min"

    def __str__(self) -> str:
        return f"Node {self.hostname} | started at {self.timestamp_str} ({self.started_ago} ago) | {self.experiment_name}"


def get_logged_runs() -> List[RunLog]:
    if not exists(RUN_LOG_PATH): return []

    with open(RUN_LOG_PATH, "rb") as run_log_file:
        run_logs: List[RunLog] = pickle.load(run_log_file)

    return run_logs


def log_run(is_global_process_0: bool, experiment_name: str) -> None:
    if not is_global_process_0: return

    run_logs = get_logged_runs()

    current_run_log = RunLog(
        experiment_name=experiment_name,
        timestamp=datetime.now(),
        hostname=gethostname(),
    )
    run_logs.append(current_run_log)

    with open(RUN_LOG_PATH, "wb") as run_log_file:
        pickle.dump(run_logs, run_log_file)


def print_most_recent_runs(num_runs: int) -> None:
    most_recent_runs = get_logged_runs()
    if not most_recent_runs:
        print("No runs found!")
        return

    most_recent_unique_runs = []
    seen = set()
    for run in reversed(most_recent_runs):  # Newest first as a stack
        if len(most_recent_unique_runs) >= num_runs: break
        if run.experiment_name in seen: continue

        most_recent_unique_runs.append(run)
        seen.add(run.experiment_name)

    # Was newest first, now we reverse to oldest first
    most_recent_unique_runs = list(reversed(most_recent_unique_runs))

    print(f"Last {len(most_recent_runs)} runs (oldest first)")
    for run in most_recent_runs:
        print(run)
