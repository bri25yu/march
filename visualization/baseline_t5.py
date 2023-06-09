from march.visualization.plot_summary import plot_comparative_experiment


plot_comparative_experiment(
    experiment_names=[
        "BaselineExperiment",
        "BaselineT5Experiment",
        "BaselineSmallFullTrainExperiment",
        "BaselineT5SmallExperiment",
        # "BaselineLargeExperiment",
        # "BaselineT5LargeExperiment",
    ],
    legend_labels=[
        "Baseline (ours)",
        "t5-base",
        "Baseline small (ours)",
        "t5-small",
        "Baseline large (ours)",
        "t5-large",
    ],
    title="Our reimplementation and t5 baseline",
    save_name="baseline_t5",
    max_steps=15_000,
)
