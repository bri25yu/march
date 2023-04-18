from typing import List

from copy import deepcopy

from march.models.baseline import *
from march.models.utils import *

from march.models.scaling_heads import ScalingHeadsEncoderBase, ScalingHeadsDecoderBase


__all__ = ["ScalingHeadsConstantTransformer", "InverseScalingHeadsConstantTransformer"]


def create_scaling_heads_constant_configs_helper(
    config: TransformerConfig, initial_num_heads: int, later_num_heads: int
) -> List[TransformerConfig]:
    layer_num_heads = \
        [initial_num_heads] * (config.num_layers // 4) \
        + [config.num_heads] * ((config.num_layers // 2) % 2) \
        + [later_num_heads] * (config.num_layers // 4)
    layer_configs = []
    for num_heads in layer_num_heads:
        config_copy = deepcopy(config)

        total_dim = config_copy.dim_model
        config_copy.num_heads = num_heads
        config_copy.dim_qkv = total_dim // num_heads

        layer_configs.append(config_copy)

    return layer_configs


def create_scaling_heads_constant_configs(config: TransformerConfig) -> List[TransformerConfig]:
    later_num_heads = config.num_heads // 2
    initial_num_heads = config.num_heads * 2
    return create_scaling_heads_constant_configs_helper(config, initial_num_heads, later_num_heads)


def create_inverse_scaling_heads_constant_configs(config: TransformerConfig) -> List[TransformerConfig]:
    later_num_heads = config.num_heads * 2
    initial_num_heads = config.num_heads // 2
    return create_scaling_heads_constant_configs_helper(config, initial_num_heads, later_num_heads)


class ScalingHeadsConstantEncoder(ScalingHeadsEncoderBase):
    def create_scaling_heads_configs(self, config: TransformerConfig) -> List[TransformerConfig]:
        return create_scaling_heads_constant_configs(config)


class ScalingHeadsConstantDecoder(ScalingHeadsDecoderBase):
    def create_scaling_heads_configs(self, config: TransformerConfig) -> List[TransformerConfig]:
        return create_scaling_heads_constant_configs(config)


class ScalingHeadsConstantTransformer(BaselineTransformer):
    ENCODER_CLS = ScalingHeadsConstantEncoder
    DECODER_CLS = ScalingHeadsConstantDecoder


class InverseScalingHeadsConstantEncoder(ScalingHeadsEncoderBase):
    def create_scaling_heads_configs(self, config: TransformerConfig) -> List[TransformerConfig]:
        return create_inverse_scaling_heads_constant_configs(config)


class InverseScalingHeadsConstantDecoder(ScalingHeadsDecoderBase):
    def create_scaling_heads_configs(self, config: TransformerConfig) -> List[TransformerConfig]:
        return create_inverse_scaling_heads_constant_configs(config)


class InverseScalingHeadsConstantTransformer(BaselineTransformer):
    ENCODER_CLS = InverseScalingHeadsConstantEncoder
    DECODER_CLS = InverseScalingHeadsConstantDecoder
