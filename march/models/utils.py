from typing import List, Tuple, Union
from torchtyping import TensorType

from abc import abstractmethod

from dataclasses import dataclass

from torch import float32, ones, rsqrt
from torch.nn import Module, Parameter

from transformers.configuration_utils import PretrainedConfig


__all__ = [
    "List",
    "Tuple",
    "Union",
    "dataclass",
    "TensorType",
    "Parameter",
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


LAYERNORM_PRECISION = float32

SequenceInputIds = TensorType["N", "L_in"]
SequenceInputEmbeds = TensorType["N", "L_in", "D"]
MultiHeadedEmbeds = TensorType["N", "H", "L_in", "D_kv"]
KeyValueStates = Tuple[MultiHeadedEmbeds, MultiHeadedEmbeds]
MultiHeadedAttention = TensorType["N", "H", "L_in", "L_out"]


@dataclass
class TransformerConfig(PretrainedConfig):
    dim_model: int = 512
    num_layers: int = 6
    dim_qkv: int = 64

    dim_feedforward: Union[None, int] = None
    num_heads: Union[None, int] = None
    dropout_prob: float = 0.1

    def __post_init__(self) -> None:
        if self.num_heads is None:
            assert self.dim_model % self.dim_qkv == 0, f"Dimensionality of the model must be divisible by dimensionality of the queries, keys, and values."
            self.num_heads = self.dim_model // self.dim_qkv

        if self.dim_feedforward is None:
            self.dim_feedforward = self.dim_model * 4


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


class LayerNorm(TransformerComponentBase):
    """
    Equivalent to T5LayerNorm.
    """
    def __init__(self, config: TransformerConfig, eps=1e-6):
        super().__init__(config)
        self.weight: TensorType["D"] = Parameter(ones(config.dim_model))
        self.variance_epsilon = eps

        self.init_weights()

    def init_weights(self) -> None:
        self.weight.data.fill_(1.0)

    def forward(self, input_embeds: SequenceInputEmbeds) -> SequenceInputEmbeds:
        input_embeds: SequenceInputEmbeds = input_embeds.to(LAYERNORM_PRECISION)
        variance: SequenceInputIds = input_embeds.pow(2).mean(-1, keepdim=True)
        input_embeds: SequenceInputEmbeds = input_embeds * rsqrt(variance + self.variance_epsilon)
        input_embeds: SequenceInputEmbeds = input_embeds.to(self.weight.dtype)

        return self.weight * input_embeds


@dataclass
class AttentionOutput:
    input_embeds: SequenceInputEmbeds
    key_value_states: Union[KeyValueStates, List[KeyValueStates]]


class AttentionBase(TransformerComponentBase):
    def __init__(self, config: TransformerConfig, is_cross_attention: bool) -> None:
        super().__init__(config)

        self.is_cross_attention = is_cross_attention

    @abstractmethod
    def forward(
        self,
        input_embeds: SequenceInputEmbeds,
        attention_mask: SequenceInputIds,
        encoder_key_value_states: KeyValueStates=None,
        encoder_attention_mask: SequenceInputIds=None,
    ) -> AttentionOutput:
        pass

    def reshape_to_head_sensitive(self, input_embeds: SequenceInputEmbeds) -> MultiHeadedEmbeds:
        config = self.config

        batch_size, sequence_length = input_embeds.size()[0], input_embeds.size()[1]
        input_embeds = input_embeds.reshape(batch_size, sequence_length, config.num_heads, config.dim_qkv)
        input_embeds = input_embeds.permute(0, 2, 1, 3)

        return input_embeds

    def reshape_to_head_insensitive(self, input_embeds: MultiHeadedEmbeds) -> SequenceInputEmbeds:
        input_embeds = input_embeds.permute(0, 2, 1, 3)
        batch_size, sequence_length = input_embeds.size()[0], input_embeds.size()[1]
        input_embeds = input_embeds.reshape(batch_size, sequence_length, -1)

        return input_embeds.contiguous()
