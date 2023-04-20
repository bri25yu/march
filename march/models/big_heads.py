from abc import abstractmethod

from march.models.baseline import *
from march.models.utils import *


__all__ = ["BigHeadsTransformer", "BigHeadsTransformerConfig"]


@dataclass
class BigHeadsTransformerConfig(TransformerConfig):
    dim_model: int = 512
    num_layers: int = 6
    dim_qkv: int = 128

    dim_feedforward: Union[None, int] = None
    num_heads: Union[None, int] = None
    dropout_prob: float = 0.1

    head_scale_size: int = 2

    def __post_init__(self) -> None:
        if self.num_heads is None:
            assert self.dim_model % self.dim_qkv == 0, f"Dimensionality of the model must be divisible by dimensionality of the queries, keys, and values."
            self.num_heads = (self.dim_model // self.dim_qkv) * self.head_scale_size

        # TODO different recombination strategy for feedforward layer? 
        if self.dim_feedforward is None:
            self.dim_feedforward = self.dim_model * 4


# TODO different recombination strategy for w_o layer? 
class BigHeadsEncoderBase(BaselineEncoder):
    @abstractmethod
    def create_scaling_heads_configs(self, config: TransformerConfig) -> List[TransformerConfig]:
        pass

    # This is a copy of `BaselineEncoder.__init__` unless specified otherwise
    def __init__(self, config: TransformerConfig) -> None:
        TransformerComponentBase.__init__(self, config)

        self.layernorms = ModuleList([LayerNorm(config) for _ in range(config.num_layers + 1)])

        ###############################
        # START Big heads Attention 
        ###############################

        # Original code:
        self.self_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(config, is_cross_attention=False) for _ in range(config.num_layers // 2)]
        )

        # New code:
        # Attention strategy now utilizes larger hidden dimension size per head, rather than 
        # heads being of size D_kv = D_model / num_heads. Now, D_kv = (D_model / num_heads) * head_scale_size
        # so the hidden dimension goes from (N, L, D_model) to (N, H, L, (D_model / num_heads ) * head_scale_size ).
        # To achieve this, 

        # layer_configs = self.create_scaling_heads_configs(config)
        # self.self_attention_layers: List[AttentionBase] = ModuleList(
        #     [self.ATTENTION_CLS(layer_config, is_cross_attention=False) for layer_config in layer_configs]
        # )

        ###############################
        # END Big heads Attention 
        ###############################

        ###############################
        # START Big heads Feed forward strategy 
        ###############################

        # Original code:
        self.feedforward_layers: List[TransformerComponentBase] = ModuleList(
            [self.FEEDFORWARD_CLS(config) for _ in range(config.num_layers // 2)]
        )

        # New code: 
        # Recombination for feed forward strategy has to go from old strategy: 
        # D_kv = D_model / num_heads 

        ###############################
        # END Big heads Feed forward strategy 
        ###############################


# class BigHeadsEncoder(ScalingHeadsEncoderBase):
#     def create_scaling_heads_configs(self, config: TransformerConfig) -> List[TransformerConfig]:
#         return create_inverse_scaling_heads_configs(config)


# class BigHeadsDecoder(ScalingHeadsDecoderBase):
#     def create_scaling_heads_configs(self, config: TransformerConfig) -> List[TransformerConfig]:
#         return create_inverse_scaling_heads_configs(config)


class BigHeadsTransformer(BaselineTransformer):
    ENCODER_CLS = BaselineEncoder
    DECODER_CLS = BaselineDecoder
