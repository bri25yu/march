from visualization import plot_comparative_experiment


plot_comparative_experiment(
    experiment_names=[
        "BaselineExperiment",
        "ScalingHeadsExperiment",
        "InverseScalingHeadsExperiment",
        "ScalingHeadsConstantExperiment",
        "InverseScalingHeadsConstantExperiment",
    ],
    legend_labels=[
        "Baseline: 12 layers each in encoder/decoder, 12 heads / 64 qkv dim",
        "First 6 layers 18 heads / 64 qkv dim, last 6 layers 6 heads / 64 qkv dim",
        "First 6 layers 6 heads / 64 qkv dim, last 6 layers 18 heads / 64 qkv dim",
        "First 6 layers 18 heads / 43 qkv dim, last 6 layers 6 heads / 128 qkv dim",
        "First 6 layers 6 heads / 128 qkv dim, last 6 layers 18 heads / 43 qkv dim",
    ],
    title="Different numbers of heads by layer",
    save_name="scaling_heads",
)
