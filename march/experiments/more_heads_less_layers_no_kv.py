from march.models.no_key_value_weights_cross_attn import NoKeyValueWeightsCrossAttentionTransformer

from march.experiments.baseline import *


class MoreHeadsLessLayersNoKVExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 2
        config.num_heads = config.num_heads + 5
        return NoKeyValueWeightsCrossAttentionTransformer(config)


class MoreHeadsLessLayersNoKV2Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 4
        config.num_heads = config.num_heads + 9
        return NoKeyValueWeightsCrossAttentionTransformer(config)


class MoreHeadsLessLayersNoKV3Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 8
        config.num_heads = config.num_heads + 19
        return NoKeyValueWeightsCrossAttentionTransformer(config)


class MoreHeadsLessLayersNoKV4Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 12
        config.num_heads = config.num_heads + 36
        return NoKeyValueWeightsCrossAttentionTransformer(config)


class MoreHeadsLessLayersNoKV5Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 16
        config.num_heads = config.num_heads + 69
        return NoKeyValueWeightsCrossAttentionTransformer(config)


class MoreHeadsLessLayersNoKV6Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 20
        config.num_heads = config.num_heads + 170
        return NoKeyValueWeightsCrossAttentionTransformer(config)


class MoreHeadsLessLayersNoKV7Experiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_layers = config.num_layers - 22
        config.num_heads = config.num_heads + 372
        return NoKeyValueWeightsCrossAttentionTransformer(config)
