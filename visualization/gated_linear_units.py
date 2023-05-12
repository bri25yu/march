from visualization import plot_comparative_experiment


plot_comparative_experiment(
    experiment_names=[
        "BaselineExperiment",
        "ReLUGatedLinearUnitExperiment",
        "GELUGatedLinearUnitExperiment",
        "SiLUGatedLinearUnitExperiment",
    ],
    legend_labels=[
        "Baseline ReLU",
        "ReLU GLU",
        "GELU GLU",
        "Swish GLU",
    ],
    title="Feedforward layers using Gated Linear Units (GLU)",
    save_name="gated_linear_units",
)
