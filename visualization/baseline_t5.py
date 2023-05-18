from visualization import plot_comparative_experiment


plot_comparative_experiment(
    experiment_names=[
        "BaselineExperiment",
        "BaselineT5Experiment",
        # "BaselineLargeExperiment",
        # "BaselineT5LargeExperiment",
    ],
    legend_labels=[
        "Baseline (ours)",
        "t5-base",
        "Baseline large (ours)",
        "t5-large",
    ],
    title="Our reimplementation and t5 baseline",
    save_name="baseline_t5",
)
