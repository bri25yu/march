from march.models.no_key_value_weights_cross_attn import NoKeyValueWeightsCrossAttentionTransformer

from march.experiments.baseline import *


class NoKeyValueWeightsCrossAttentionExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return NoKeyValueWeightsCrossAttentionTransformer(config)


class NoKeyValueWeightsCrossAttentionWithExtraHeadsExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.num_heads += 2
        return NoKeyValueWeightsCrossAttentionTransformer(config)


class NoKeyValueWeightsCrossAttentionWithExtraDimExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        config.dim_model += config.num_heads * 2
        config.dim_qkv += 2
        return NoKeyValueWeightsCrossAttentionTransformer(config)
