from dataclasses import dataclass

from numpy import arange, exp, ndarray, power

from pandas import DataFrame

from matplotlib.axes import Axes

from scipy.optimize import least_squares


@dataclass
class ScalingLawForCompute:
    steps: ndarray
    loss: ndarray
    legend_label: str
    stats_df: DataFrame

    # From 10k to 1mil steps i.e. 10 ** 4 to 10 ** 6
    TOTAL_TRAJECTORY = (10 ** arange(4.0, 6.0 + 0.01, step=0.01)).astype(int)
    MODELING_RANGE = (12_000, 15_000)  # Steps. TODO make this dynamic?

    def __post_init__(self) -> None:
        self.fit_scaling_law_for_compute()

    @staticmethod
    def fit_function(t: ndarray, a: int, b: int, c: int) -> ndarray:
        # TODO improve function
        return a * power(t, -b) + c

    def fit_scaling_law_for_compute(self) -> None:
        low, high = self.MODELING_RANGE
        in_range = (low <= self.steps) & (self.steps <= high)
        steps = self.steps[in_range]
        loss = self.loss[in_range]

        def residual_fn(guess: ndarray) -> ndarray:
            return self.fit_function(steps, *guess) - loss

        self.fit_params = least_squares(
            fun=residual_fn,
            x0=(10, 0.5, 2.0),
            loss="soft_l1",
        ).x

        def get_val_str(step):
            predicted_losses = self.fit_function(step, *self.fit_params)
            predicted_perplexity = exp(predicted_losses)
            return f"{predicted_perplexity:.3f}"

        a, b, c = self.fit_params
        self.stats_df.loc[len(self.stats_df.index)] = {
            "Experiment": self.legend_label,
            "Scaling law": f"{a:.2f}(t ** -{b:.3f}) - {c:.2f}",
            "PPL at 100k": get_val_str(100_000),
            "PPL at 300k": get_val_str(300_000),
            "PPL at 1M": get_val_str(1_000_000),
        }

    def plot_over_all(self, ax: Axes) -> None:
        predicted_values = self.fit_function(self.TOTAL_TRAJECTORY, *self.fit_params)

        legend_label_full = f"{self.legend_label}"
        ax.plot(self.TOTAL_TRAJECTORY, predicted_values, label=legend_label_full)
