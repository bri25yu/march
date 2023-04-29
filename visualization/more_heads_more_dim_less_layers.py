from visualization import plot_comparative_experiment


plot_comparative_experiment(
    experiment_names=[
        "BaselineExperiment",
        "MoreHeadsMoreDimLessLayersExperiment",
        "MoreHeadsMoreDimLessLayers2Experiment",
        "MoreHeadsMoreDimLessLayers3Experiment",
        "MoreHeadsMoreDimLessLayers4Experiment",
        "MoreHeadsMoreDimLessLayers5Experiment",
    ],
    legend_labels=[
        "Baseline: 24 layers, 768 d_model, 12 heads",
        "18 layers, 768 + 64 * 1 = 832 d_model, 12 + 1 + 4 = 17 heads",
        "14 layers, 768 + 64 * 2 = 896 d_model, 12 + 2 + 8 = 22 heads",
        "10 layers, 768 + 64 * 3 = 960 d_model, 12 + 3 + 15 = 30 heads",
        "8 layers, 768 + 64 * 4 = 1024 d_model, 12 + 4 + 20 = 36 heads",
        "4 layers, 768 + 64 * 8 = 1280 d_model, 12 + 8 + 38 = 58 heads",
    ],
    title="More dim model and more heads less layers",
    save_name="more_heads_more_dim_less_layers",
)
