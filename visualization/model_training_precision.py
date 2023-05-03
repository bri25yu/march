from visualization import plot_comparative_experiment


plot_comparative_experiment(
    experiment_names=[
        "BaselineExperiment",
        "BaselineFP32Experiment",
    ],
    legend_labels=[
        "Baseline (BF16)",
        "Baseline (FP32)",
    ],
    title="Model training precision",
    save_name="model_training_precision",
)
