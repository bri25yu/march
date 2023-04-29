from visualization import plot_comparative_experiment


plot_comparative_experiment(
    experiment_names=[
        "BaselineExperiment",
        "MoreDimLessLayersExperiment",
        "MoreDimLessLayers2Experiment",
        "MoreDimLessLayers3Experiment",
        "MoreDimLessLayers4Experiment",
        "MoreDimLessLayers5Experiment",
    ],
    legend_labels=[
        "Baseline: 24 layers, 768 d_model, 12 heads",
        "20 layers, 768 + 64 = 832 d_model, 12 + 1 = 13 heads",
        "18 layers, 768 + 64 * 2 = 896 d_model, 12 + 2 = 14 heads",
        "14 layers, 768 + 64 * 4 = 1024 d_model, 12 + 4 = 16 heads",
        "8 layers, 768 + 64 * 11 = 1472 d_model, 12 + 11 = 23 heads",
        "4 layers, 768 + 64 * 20 = 2048 d_model, 12 + 20 = 32 heads",
    ],
    title="More dim model (resulting in more heads) less layers",
    save_name="more_dim_less_layers",
)
