from typing import List

from copy import deepcopy

from torch.nn import ModuleList

from march.models.baseline import *
from march.models.utils import *


__all__ = ["ScalingHeadsTransformer"]


def create_scaling_heads_configs(config: TransformerConfig) -> List[TransformerConfig]:
    later_num_heads = config.num_heads // 2
    initial_num_heads = config.num_heads + later_num_heads
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


class ScalingHeadsEncoder(BaselineEncoder):
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

        layer_configs = create_scaling_heads_configs(config)
        self.self_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(layer_config, is_cross_attention=False) for layer_config in layer_configs]
        )

        ###############################
        # END Scaling heads
        ###############################

        self.feedforward_layers: List[TransformerComponentBase] = ModuleList(
            [self.FEEDFORWARD_CLS(config) for _ in range(config.num_layers // 2)]
        )


class ScalingHeadsDecoder(BaselineDecoder):
    # This is a copy of `BaselineDecoder.__init__` unless specified otherwise
    def __init__(self, config: TransformerConfig) -> None:
        TransformerComponentBase().__init__(self, config)

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

        layer_configs = create_scaling_heads_configs(config)
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


class ScalingHeadsTransformer(BaselineTransformer):
    ENCODER_CLS = ScalingHeadsEncoder
    DECODER_CLS = ScalingHeadsDecoder
