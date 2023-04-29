from visualization import plot_comparative_experiment


plot_comparative_experiment(
    experiment_names=[
        "BaselineExperiment",
        "NoKeyValueWeightsCrossAttentionExperiment",
        "NoKeyValueWeightsCrossAttentionWithExtraHeadsExperiment",
    ],
    legend_labels=[
        "Baseline",
        "No cross-attn weights on encoder keys/values",
        "No cross-attn weights on encoder keys/values with extra heads",
    ],
    title="Removing cross attention weights on encoder keys/values",
    save_name="no_cross_attention_kv_weights",
)
