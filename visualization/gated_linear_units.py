from visualization import plot_comparative_experiment


plot_comparative_experiment(
    experiment_names=[
        "BaselineExperiment",
        "ReLUGatedLinearUnitExperiment",
        "GELUGatedLinearUnitExperiment",
        "SiLUGatedLinearUnitExperiment",
    ],
    legend_labels=[
        "Baseline ReLU intermediate activation",
        "ReLU GLU (ReGLU)",
        "GELU GLU (GEGLU)",
        "Swish (SiLU) GLU (SwiGLU)",
    ],
    title="Feedforward layers using Gated Linear Units (GLU)",
    save_name="gated_linear_units",
)
