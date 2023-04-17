from typing import List

from abc import abstractmethod

from copy import deepcopy

from torch.nn import ModuleList

from march.models.baseline import *
from march.models.utils import *


__all__ = ["ScalingHeadsTransformer", "InverseScalingHeadsTransformer"]


def create_scaling_heads_configs_helper(
    config: TransformerConfig, initial_num_heads: int, later_num_heads: int
) -> List[TransformerConfig]:
    layer_num_heads = \
        [initial_num_heads] * (config.num_layers // 4) \
        + [config.num_heads] * ((config.num_layers // 2) % 2) \
        + [later_num_heads] * (config.num_layers // 4)
    layer_configs = []
    for num_heads in layer_num_heads:
        config_copy = deepcopy(config)
        config_copy.num_heads = num_heads
        layer_configs.append(config_copy)

    return layer_configs


def create_scaling_heads_configs(config: TransformerConfig) -> List[TransformerConfig]:
    later_num_heads = config.num_heads // 2
    initial_num_heads = config.num_heads + later_num_heads
    return create_scaling_heads_configs_helper(config, initial_num_heads, later_num_heads)


def create_inverse_scaling_heads_configs(config: TransformerConfig) -> List[TransformerConfig]:
    initial_num_heads = config.num_heads // 2
    later_num_heads = config.num_heads + initial_num_heads
    return create_scaling_heads_configs_helper(config, initial_num_heads, later_num_heads)


class ScalingHeadsEncoderBase(BaselineEncoder):
    @abstractmethod
    def create_scaling_heads_configs(self, config: TransformerConfig) -> List[TransformerConfig]:
        pass

    # This is a copy of `BaselineEncoder.__init__` unless specified otherwise
    def __init__(self, config: TransformerConfig) -> None:
        TransformerComponentBase.__init__(self, config)

        self.layernorms = ModuleList([LayerNorm(config) for _ in range(config.num_layers + 1)])

        ###############################
        # START Scaling heads
        ###############################

        # Original code:
        # self.self_attention_layers: List[AttentionBase] = ModuleList(
        #     [self.ATTENTION_CLS(config, is_cross_attention=False) for _ in range(config.num_layers // 2)]
        # )

        layer_configs = self.create_scaling_heads_configs(config)
        self.self_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(layer_config, is_cross_attention=False) for layer_config in layer_configs]
        )

        ###############################
        # END Scaling heads
        ###############################

        self.feedforward_layers: List[TransformerComponentBase] = ModuleList(
            [self.FEEDFORWARD_CLS(config) for _ in range(config.num_layers // 2)]
        )


class ScalingHeadsDecoderBase(BaselineDecoder):
    @abstractmethod
    def create_scaling_heads_configs(self, config: TransformerConfig) -> List[TransformerConfig]:
        pass

    # This is a copy of `BaselineDecoder.__init__` unless specified otherwise
    def __init__(self, config: TransformerConfig) -> None:
        TransformerComponentBase.__init__(self, config)

        self.layernorms = ModuleList([LayerNorm(config) for _ in range(((config.num_layers // 2) * 3) + 1)])

        ###############################
        # START Scaling heads
        ###############################

        # Original code:
        # self.self_attention_layers: List[AttentionBase] = ModuleList(
        #     [self.ATTENTION_CLS(config, is_cross_attention=False) for _ in range(config.num_layers // 2)]
        # )
        # self.cross_attention_layers: List[AttentionBase] = ModuleList(
        #     [self.ATTENTION_CLS(config, is_cross_attention=True) for _ in range(config.num_layers // 2)]
        # )

        layer_configs = self.create_scaling_heads_configs(config)
        self.self_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(layer_config, is_cross_attention=False) for layer_config in layer_configs]
        )
        self.cross_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(layer_config, is_cross_attention=True) for layer_config in layer_configs]
        )

        ###############################
        # END Scaling heads
        ###############################

        self.feedforward_layers: List[TransformerComponentBase] = ModuleList(
            [self.FEEDFORWARD_CLS(config) for _ in range(config.num_layers // 2)]
        )


class ScalingHeadsEncoder(ScalingHeadsEncoderBase):
    def create_scaling_heads_configs(self, config: TransformerConfig) -> List[TransformerConfig]:
        return create_scaling_heads_configs(config)


class ScalingHeadsDecoder(ScalingHeadsDecoderBase):
    def create_scaling_heads_configs(self, config: TransformerConfig) -> List[TransformerConfig]:
        return create_scaling_heads_configs(config)


class ScalingHeadsTransformer(BaselineTransformer):
    ENCODER_CLS = ScalingHeadsEncoder
    DECODER_CLS = ScalingHeadsDecoder


class InverseScalingHeadsEncoder(ScalingHeadsEncoderBase):
    def create_scaling_heads_configs(self, config: TransformerConfig) -> List[TransformerConfig]:
        return create_inverse_scaling_heads_configs(config)


class InverseScalingHeadsDecoder(ScalingHeadsDecoderBase):
    def create_scaling_heads_configs(self, config: TransformerConfig) -> List[TransformerConfig]:
        return create_inverse_scaling_heads_configs(config)


class InverseScalingHeadsTransformer(BaselineTransformer):
    ENCODER_CLS = InverseScalingHeadsEncoder
    DECODER_CLS = InverseScalingHeadsDecoder
