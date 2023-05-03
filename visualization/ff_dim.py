from visualization import plot_comparative_experiment


plot_comparative_experiment(
    experiment_names=[
        "FFDimOctupleFromLayersExperiment",
        "FFDimOctupleFromDimExperiment",
        "BaselineExperiment",
        "FFDimDoubleToLayersExperiment",
        "FFDimDoubleToDimExperiment",
        "FFDimSameToLayersExperiment",
        "FFDimSameToDimExperiment",
        "FFDimHalfToLayersExperiment",
        "FFDimHalfToDimExperiment",
    ],
    legend_labels=[
        "14 layers, 8.0 feedforward scale, 768 model dim",
        "24 layers, 8.0 feedforward scale, 768 - 64 * 3 = 576 model dim",
        "Baseline: 24 layers, 4.0 feedforward scale, 768 model dim",
        "32 layers, 2.0 feedforward scale, 768 model dim",
        "24 layers, 2.0 feedforward scale, 768 + 64 * 2 = 896 model dim",
        "42 layers, 1.0 feedforward scale, 768 model dim",
        "24 layers, 1.0 feedforward scale, 768 + 64 * 3 = 960 model dim",
        "48 layers, 0.5 feedforward scale, 768 model dim",
        "24 layers, 0.5 feedforward scale, 768 + 64 * 4 = 1024 model dim",
    ],
    title="Feedforward dimension",
    save_name="ffdim",
)
