from typing import List, Tuple, Union
from torchtyping import TensorType

from abc import abstractmethod

from dataclasses import dataclass

from torch import FloatTensor, float32, rsqrt, ones
from torch.nn import Module, Parameter
from torch.nn.functional import dropout

from transformers.configuration_utils import PretrainedConfig
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from march.datasets.c4 import VOCAB_SIZE


__all__ = [
    "List",
    "Tuple",
    "Union",
    "dataclass",
    "TensorType",
    "Parameter",
    "FloatTensor",
    "float32",
    "SequenceInputIds",
    "SequenceInputEmbeds",
    "KeyValueStates",
    "MultiHeadedEmbeds",
    "MultiHeadedAttention",
    "TransformerConfig",
    "TransformerComponentBase",
    "LayerNorm",
    "AttentionOutput",
    "AttentionBase",
]


SequenceInputIds = TensorType["N", "L_in"]
SequenceInputEmbeds = TensorType["N", "L_in", "D"]
MultiHeadedEmbeds = TensorType["N", "H", "L_in", "D_kv"]
KeyValueStates = Tuple[MultiHeadedEmbeds, MultiHeadedEmbeds]
MultiHeadedAttention = TensorType["N", "H", "L_in", "L_out"]


@dataclass
class TransformerConfig(PretrainedConfig):
    dim_model: int = 768
    num_layers: int = 24
    dim_qkv: int = 64

    feedforward_scale: Union[float, int] = 4
    dim_feedforward: Union[None, int] = None
    num_heads: Union[None, int] = None
    dropout_prob: float = 0.1
    vocab_size: int = VOCAB_SIZE

    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128

    def __post_init__(self) -> None:
        if self.num_heads is None:
            assert self.dim_model % self.dim_qkv == 0, f"Dimensionality of the model must be divisible by dimensionality of the queries, keys, and values."
            self.num_heads = self.dim_model // self.dim_qkv

        if self.dim_feedforward is None:
            self.dim_feedforward = int(self.dim_model * self.feedforward_scale)


class TransformerComponentBase(Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

        self.config = config

    @abstractmethod
    def init_weights(self) -> None:
        pass

    @abstractmethod
    def forward(self, input_embeds: SequenceInputEmbeds) -> SequenceInputEmbeds:
        pass

    def apply_residual(self, residual: SequenceInputEmbeds, current: SequenceInputEmbeds) -> SequenceInputEmbeds:
        return residual + self.apply_dropout(current)

    def apply_dropout(self, input_embeds: SequenceInputEmbeds) -> SequenceInputEmbeds:
        dropout_prob = self.config.dropout_prob
        return dropout(input_embeds, p=dropout_prob, training=self.training)


class LayerNorm(TransformerComponentBase):
    """
    Equivalent to T5LayerNorm.
    """
    def __init__(self, config: TransformerConfig, eps=1e-6):
        super().__init__(config)

        self.weight: TensorType["D"] = Parameter(ones(config.dim_model,))
        self.variance_epsilon = eps

    def forward(self, input_embeds: SequenceInputEmbeds) -> SequenceInputEmbeds:
        variance: SequenceInputIds = input_embeds.to(float32).pow(2).mean(-1, keepdim=True)
        input_embeds: SequenceInputEmbeds = input_embeds * rsqrt(variance + self.variance_epsilon)
        input_embeds: SequenceInputEmbeds = input_embeds.to(self.weight.dtype)

        return self.weight * input_embeds


ALL_LAYERNORM_LAYERS.append(LayerNorm)


@dataclass
class AttentionOutput:
    input_embeds: SequenceInputEmbeds
    position_bias: MultiHeadedAttention


class AttentionBase(TransformerComponentBase):
    def __init__(self, config: TransformerConfig, is_cross_attention: bool, is_decoder: bool) -> None:
        super().__init__(config)

        self.is_cross_attention = is_cross_attention
        self.is_decoder = is_decoder

    @abstractmethod
    def forward(
        self,
        input_embeds: SequenceInputEmbeds,
        attention_mask: SequenceInputIds,
        position_bias: MultiHeadedAttention=None,
        encoder_hidden_state: SequenceInputEmbeds=None,
    ) -> AttentionOutput:
        pass

    def reshape_to_head_sensitive(self, input_embeds: SequenceInputEmbeds) -> MultiHeadedEmbeds:
        config = self.config

        batch_size, sequence_length = input_embeds.size()[0], input_embeds.size()[1]

        # Input embeds reshape to shape (N, L, H, D_kv) from (N, L, (H * D_kv))
        input_embeds = input_embeds.reshape(batch_size, sequence_length, config.num_heads, config.dim_qkv)
        # Permute to shape (N, H, L, D_kv)
        input_embeds = input_embeds.permute(0, 2, 1, 3)

        return input_embeds

    def reshape_to_head_insensitive(self, input_embeds: MultiHeadedEmbeds) -> SequenceInputEmbeds:
        # input_embeds input format (batch_size, num_heads, sequence_length, dim_qkv) (N, H, L, D_kv)
        # permute to (N, L, H, D_kv)
        input_embeds = input_embeds.permute(0, 2, 1, 3)
        batch_size, sequence_length = input_embeds.size()[0], input_embeds.size()[1]
        # reshape to (N, L, (H * D_kv))
        input_embeds = input_embeds.reshape(batch_size, sequence_length, -1)
        return input_embeds.contiguous()
