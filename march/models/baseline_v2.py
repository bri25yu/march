from typing import Tuple, Optional
from torchtyping import TensorType

from dataclasses import dataclass

from torch import arange, bfloat16, cat, embedding, empty, float32, outer
from torch.nn import Module, ModuleList, Parameter
from torch.nn.functional import cross_entropy, linear
from torch.jit import script

from xformers.ops import LowerTriangularMask, memory_efficient_attention, swiglu, unbind

from transformers.utils.import_utils import is_torch_bf16_gpu_available
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import PretrainedConfig

from apex.normalization import FusedRMSNorm
ALL_LAYERNORM_LAYERS.append(FusedRMSNorm)

from march.datasets.c4 import MAX_LENGTH, VOCAB_SIZE

from march.models.utils import TransformerComponentBase


D = TensorType["D"]
HalfD = TensorType["D/2"]
L = TensorType["L"]
LHalfD = TensorType["L", "D/2"]
LD = TensorType["L", "2D"]
NL = TensorType["N", "L"]
NLD = TensorType["N", "L", "D"]
NLL = TensorType["N", "L", "L"]
NHLL = TensorType["N", "H", "L_q", "L_k"]
NLHDkv = TensorType["N", "L", "H", "Dkv"]


@dataclass
class BaselineV2Config(PretrainedConfig):
    dim_model: int = 896
    num_decoder_layers: int = 8
    dim_qkv: int = 64

    feedforward_scale: float = (2 / 3) * 4

    vocab_size: int = VOCAB_SIZE
    dtype = bfloat16 if is_torch_bf16_gpu_available() else float32
    max_length: int = MAX_LENGTH

    def __post_init__(self) -> None:
        assert (
            self.dim_model % self.dim_qkv == 0
        ), f"Dimensionality of the model must be divisible by dimensionality of the queries, keys, and values."
        self.num_heads = self.dim_model // self.dim_qkv

        base = 64  # Round to multiple of 64 for better performance
        self.dim_feedforward = (
            round(self.dim_model * self.feedforward_scale / base) * base
        )


class Linear(TransformerComponentBase):
    def __init__(
        self,
        config: BaselineV2Config,
        in_features: int,
        out_features: int,
        std: Optional[float] = None,
    ) -> None:
        super().__init__(config)

        self.weight = Parameter(empty((out_features, in_features), dtype=config.dtype))
        self.weight.data.normal_(mean=0.0, std=std or in_features**-0.5)

    def forward(self, embeds: NLD) -> NLD:
        return linear(embeds, self.weight)


@script
def apply_rotary_pos_emb(embeds, cos, sin):
    # embeds is NLHDkv, cos and sin are 1L1D. output is NLHDkv
    # Handle a possible sequence length mismatch in between q and k
    L = embeds.size(1)
    cos = cos[:, :L, :, :]
    sin = sin[:, :L, :, :]

    # left_half and right_half are NLHHalfDkv. embeds_half_rotated is NLHDkv
    left_half, right_half = embeds.chunk(2, dim=3)  # In the D dimension
    embeds_half_rotated = cat((-right_half, left_half), dim=3)

    return embeds * cos + embeds_half_rotated * sin


# Copied and reformatted from xformers
class RotaryEmbedding(TransformerComponentBase):
    def __init__(self, config: BaselineV2Config) -> None:
        super().__init__(config)

        inv_freq: HalfD = 1.0 / (10000 ** (arange(0, config.dim_model, 2, dtype=config.dtype) / config.dim_model))
        self.register_buffer("inv_freq", inv_freq)
        self.initialize_cos_sin_tables(config.max_length)

    def initialize_cos_sin_tables(self, seq_len: int) -> None:
        dtype = self.config.dtype

        t: L = arange(seq_len, dtype=dtype)
        freqs: LHalfD = outer(t, self.inv_freq)
        emb: LD = cat((freqs, freqs), dim=-1)

        # sin and cos cached are 1L1D
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :], persistent=False)
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :], persistent=False)

    def forward(self, query: NLHDkv, key: NLHDkv) -> Tuple[NLHDkv, NLHDkv]:
        max_length = self.config.max_length
        L_q, L_k = query.size(1), key.size(1)

        if L_q > max_length or L_k > max_length:
            self.initialize_cos_sin_tables(max(L_q, L_k))

        return (
            apply_rotary_pos_emb(query, self.cos_cached, self.sin_cached),
            apply_rotary_pos_emb(key, self.cos_cached, self.sin_cached),
        )


class BaselineV2Attention(TransformerComponentBase):
    def __init__(self, config: BaselineV2Config, is_causal: bool=False) -> None:
        super().__init__(config)

        self.attn_bias = LowerTriangularMask() if is_causal else None

        D, HDkv = config.dim_model, config.num_heads * config.dim_qkv
        self.w_q = Linear(config, D, HDkv)
        self.w_k = Linear(config, D, HDkv)
        self.w_v = Linear(config, D, HDkv)
        self.w_o = Linear(config, HDkv, D)
        self.rotary_emb = RotaryEmbedding(config.dim_qkv)

    def forward(self, embeds: NLD, encoder_embeds: Optional[NLD] = None) -> NLD:
        config = self.config
        N, L_q, D = embeds.size()
        H, Dkv = config.num_heads, config.dim_qkv

        if encoder_embeds is not None:  # cross attention
            L_k = encoder_embeds.size(1)
            key_value_embeds = encoder_embeds
        else:  # self attention
            L_k = L_q
            key_value_embeds = embeds

        query: NLHDkv = self.w_q(embeds).view(N, L_q, H, Dkv)
        key: NLHDkv = self.w_k(key_value_embeds).view(N, L_k, H, Dkv)
        value: NLHDkv = self.w_v(key_value_embeds).view(N, L_k, H, Dkv)

        query, key = self.rotary_emb(query, key)

        attention_values: NLHDkv = memory_efficient_attention(query, key, value, self.attn_bias)

        attention_output: NLD = self.w_o(attention_values.reshape(N, L_q, D))
        return attention_output


class SwiGLU(TransformerComponentBase):
    def __init__(self, config: BaselineV2Config) -> None:
        super().__init__(config)

        D, Dff = config.dim_model, config.dim_feedforward
        self.w12 = Linear(config, D, 2 * Dff)
        self.w3 = Linear(config, Dff, D)

    def forward(self, embeds: NLD) -> NLD:
        D, Dff = self.config.dim_model, self.config.dim_feedforward
        w1, w2 = unbind(self.w12.weight.view([2, Dff, D]), dim=0)
        return swiglu(embeds, w1, None, w2, None, self.w3.weight, None)


class BaselineV2EncoderDecoder(TransformerComponentBase):
    ATTENTION_CLS = BaselineV2Attention
    FEEDFORWARD_CLS = SwiGLU

    def __init__(self, config: BaselineV2Config, is_decoder: bool) -> None:
        super().__init__(config)

        self.is_decoder = is_decoder

        attn_cls = self.ATTENTION_CLS
        ff_cls = self.FEEDFORWARD_CLS
        layernorm_fn = lambda: FusedRMSNorm(config.dim_model, eps=1e-6)

        self.layers = ModuleList()
        for _ in range(config.num_decoder_layers):
            self.layers.append(layer := Module())

            layer.selfattn_ln, layer.selfattn = layernorm_fn(), attn_cls(config, is_causal=is_decoder)

            if is_decoder:
                layer.crossattn_ln, layer.crossattn = layernorm_fn(), attn_cls(config)

            layer.ff_ln, layer.ff = layernorm_fn(), ff_cls(config)

        self.final_layernorm = layernorm_fn()

    def forward(self, embeds: NLD, encoder_embeds: Optional[NLD] = None) -> NLD:
        for layer in self.layers:
            embeds = embeds + layer.selfattn(layer.selfattn_ln(embeds))

            if self.is_decoder:
                embeds = embeds + layer.crossattn(layer.crossattn_ln(embeds), encoder_embeds)

            embeds = embeds + layer.ff(layer.ff_ln(embeds))

        embeds = self.final_layernorm(embeds)
        return embeds


class BaselineV2Transformer(TransformerComponentBase):
    ENCODERDECODER_CLS = BaselineV2EncoderDecoder

    def __init__(self, config: BaselineV2Config) -> None:
        super().__init__(config)

        self.embedding = Linear(config, config.dim_model, config.vocab_size, std=1.0)
        self.encoder = self.ENCODERDECODER_CLS(config, is_decoder=False)
        self.decoder = self.ENCODERDECODER_CLS(config, is_decoder=True)

    def forward(self, input_ids: NL, decoder_input_ids: NL, labels: NL) -> Seq2SeqLMOutput:
        config = self.config

        encoder_embeds: NLD = self.encoder(embedding(self.embedding.weight, input_ids))

        decoder_embeds: NLD = embedding(self.embedding.weight, decoder_input_ids)
        decoder_embeds: NLD = self.decoder(decoder_embeds, encoder_embeds)

        logits: NLD = self.embedding(decoder_embeds * (config.dim_model**-0.5))

        loss = cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(loss=loss, logits=logits)
