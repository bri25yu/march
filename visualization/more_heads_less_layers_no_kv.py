from visualization import plot_comparative_experiment


plot_comparative_experiment(
    experiment_names=[
        "BaselineExperiment",
        "MoreHeadsLessLayersNoKVExperiment",
        "MoreHeadsLessLayersNoKV2Experiment",
        "MoreHeadsLessLayersNoKV3Experiment",
        "MoreHeadsLessLayersNoKV4Experiment",
        "MoreHeadsLessLayersNoKV5Experiment",
        "MoreHeadsLessLayersNoKV6Experiment",
    ],
    legend_labels=[
        "Baseline: 24 layers, 12 heads",
        "No KV, 22 layers, 17 heads",
        "No KV, 20 layers, 21 heads",
        "No KV, 16 layers, 31 heads",
        "No KV, 12 layers, 48 heads",
        "No KV, 8 layers, 81 heads",
        "No KV, 4 layers, 182 heads",
    ],
    title="More heads less layers with no cross-attention key/value weights",
    save_name="more_heads_less_layers_no_kv",
)
