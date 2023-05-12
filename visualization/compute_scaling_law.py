from dataclasses import dataclass

from numpy import arange, exp, log, ndarray, power

from pandas import DataFrame

from matplotlib.axes import Axes

from scipy.optimize import brute


@dataclass
class ScalingLawForCompute:
    steps: ndarray
    eval_loss: ndarray
    legend_label: str
    stats_df: DataFrame

    # From 10 to 1mil steps i.e. 10 ** 1 to 10 ** 6
    TOTAL_TRAJECTORY = (10 ** arange(3.0, 6.0 + 0.01, step=0.01)).astype(int)
    START_MODELING_STEP = 100  # 0 uses all steps

    def __post_init__(self) -> None:
        past_initial_curve = self.steps >= self.START_MODELING_STEP
        self.steps = self.steps[past_initial_curve]
        self.eval_loss = self.eval_loss[past_initial_curve]

        # L(t) = a * (t ** b) + c
        # From https://arxiv.org/abs/2010.14701
        self.fit_function = lambda t, a, b, c: a * power(t / 100, -b) - c

        self.fit_scaling_law_for_compute()

    def fit_scaling_law_for_compute(self) -> None:
        # TODO Fix power law fitting logic
        # The curve fit must qualitatively pass through the majority of the curve
        # You can subjectively tell when its good or not. 

        def loss_fn(params):
            y_hat = self.fit_function(self.steps, *params)
            y = self.eval_loss
            abs_diff = abs(y - y_hat)

            # Cauchy loss rho(z) = ln(1 + z)
            loss_unreduced = log(1 + abs_diff)

            loss = loss_unreduced.mean()

            return loss

        ranges = [
            (8.0, 12.0),
            (0.01, 0.10),
            (2, 4),
        ]
        self.fit_params = brute(
            loss_fn,
            ranges=ranges,
            Ns=50,
            finish=None,
            full_output=True,
        )[0]
        residual = abs(self.eval_loss - self.fit_function(self.steps, *self.fit_params)).mean()

        def get_val_str(step):
            predicted_losses = self.fit_function(step, *self.fit_params)
            predicted_perplexity = exp(predicted_losses)
            return f"{predicted_perplexity:.3f}"

        a, b, c = self.fit_params
        self.stats_df.loc[len(self.stats_df.index)] = {
            "Experiment": self.legend_label,
            "Scaling law": f"{a:.2f}(t ** -{b:.3f}) - {c:.2f}",
            "Mean L1 residual": f"{residual:.3f}",
            "PPL at 1k": get_val_str(1_000),
            "PPL at 10k": get_val_str(10_000),
            "PPL at 100k": get_val_str(100_000),
            "PPL at 300k": get_val_str(300_000),
            "PPL at 1M": get_val_str(1_000_000),
        }

    def plot_over_steps(self, ax: Axes, color: str) -> None:
        predicted_values = self.fit_function(self.steps, *self.fit_params)
        legend_label_full = f"Predicted {self.legend_label}"
        ax.plot(self.steps, predicted_values, label=legend_label_full, color=color, linestyle="dashed")

    def plot_over_all(self, ax: Axes) -> None:
        predicted_values = exp(self.fit_function(self.TOTAL_TRAJECTORY, *self.fit_params))
        # predicted_values = self.fit_function(self.TOTAL_TRAJECTORY, *self.fit_params)
        # print(predicted_values.max(), predicted_values.min())

        legend_label_full = f"{self.legend_label}"
        ax.plot(self.TOTAL_TRAJECTORY, predicted_values, label=legend_label_full)
