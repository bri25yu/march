from visualization import plot_comparative_experiment


plot_comparative_experiment(
    experiment_names=[
        "BaselineExperiment",
        "MoreHeadsLessLayersExperiment",
        "MoreHeadsLessLayers2Experiment",
        "MoreHeadsLessLayers3Experiment",
        "MoreHeadsLessLayers4Experiment",
        "MoreHeadsLessLayers5Experiment",
        "MoreHeadsLessLayers6Experiment",
    ],
    legend_labels=[
        "Baseline: 24 layers, 12 heads",
        "22 layers, 14 heads",
        "20 layers, 16 heads",
        "16 layers, 22 heads",
        "12 layers, 31 heads",
        "8 layers, 46 heads",
        "4 layers, 78 heads",
    ],
    title="More heads less layers",
    save_name="more_heads_less_layers",
)
