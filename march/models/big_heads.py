from abc import abstractmethod

from march.models.baseline import *
from march.models.utils import *


__all__ = ["BigHeadsTransformer", "BigHeadsTransformerConfig"]


@dataclass
class BigHeadsTransformerConfig(TransformerConfig):
    dim_qkv: Union[None, int] = None

    # Fix num_heads, since dim_qkv is not fixed
    num_heads: int = 12

    head_scale_size: int = 2

    dim_w_o_output_size: Union[None, int] = None
    # dim_w_o_output_size :=  num_heads * dim_qkv / dim_w_o_output_scaling
    dim_w_o_output_scaling: Union[int, float] = 1

    def __post_init__(self) -> None:
        if self.dim_qkv is None:
            self.dim_qkv = self.dim_model * self.head_scale_size // self.num_heads
        
        assert(self.dim_model % self.dim_qkv == 0)

        if self.dim_feedforward is None:
            self.dim_feedforward = self.dim_model * self.feedforward_scale

        if self.dim_w_o_output_size is None:
            self.dim_w_o_output_size = self.num_heads * self.dim_qkv
        else:
            self.dim_w_o_output_size = int(self.num_heads * self.dim_qkv // self.dim_w_o_output_scaling)


class BigHeadsAttention(BaselineAttention):
    def __init__(self, config: BigHeadsTransformerConfig, is_cross_attention: bool) -> None:
        super().__init__(config, is_cross_attention)

        self.w_q = Linear(config.dim_model, config.num_heads * config.dim_qkv, bias=False)

        # START big heads attention
        # Make w_o actually a linear layer and then downcast later in the feedforward layer
        # self.w_o = Linear(config.num_heads * config.dim_qkv, config.dim_model, bias=False)
        self.w_o = Linear(config.num_heads * config.dim_qkv, config.dim_w_o_output_size, bias=False)
        
        # END big heads attention

        if self.is_cross_attention:
            self.w_k = Linear(config.num_heads * config.dim_qkv, config.num_heads * config.dim_qkv, bias=False)
            self.w_v = Linear(config.num_heads * config.dim_qkv, config.num_heads * config.dim_qkv, bias=False)
        else:
            self.w_k = Linear(config.dim_model, config.num_heads * config.dim_qkv, bias=False)
            self.w_v = Linear(config.dim_model, config.num_heads * config.dim_qkv, bias=False)


class BigHeadsFeedforward(BaselineFeedforward):
    def __init__(self, config: BigHeadsTransformerConfig) -> None:
        super().__init__(config)

        # Keep the dimension from output of w_o, and if necessary still cast up to 4D. Otherwise it acts as a FF layer. 
        # self.up_projection = Linear(config.dim_model, config.dim_feedforward, bias=False)
        self.up_projection = Linear(config.dim_w_o_output_size, config.dim_feedforward, bias=False)

        # Still project from dim_feedforward (normally 4D) to D
        self.down_projection = Linear(config.dim_feedforward, config.dim_model, bias=False)


class BigHeadsEncoder(EncoderBase):
    ATTENTION_CLS = BigHeadsAttention
    FEEDFORWARD_CLS = BigHeadsFeedforward


class BigHeadsDecoder(DecoderBase):
    ATTENTION_CLS = BigHeadsAttention
    FEEDFORWARD_CLS = BigHeadsFeedforward


class BigHeadsTransformer(BaselineTransformer):
    ENCODER_CLS = BigHeadsEncoder
    DECODER_CLS = BigHeadsDecoder
