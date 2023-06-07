from typing import Optional, Union
from torchtyping import TensorType

from dataclasses import dataclass

from torch import bfloat16, embedding, empty, finfo, float32
from torch.nn import Module, ModuleList, Parameter
from torch.nn.functional import cross_entropy, linear

from xformers.components.positional_embedding import RotaryEmbedding
from xformers.ops import memory_efficient_attention, swiglu, unbind

from transformers.utils.import_utils import is_torch_bf16_gpu_available
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import PretrainedConfig

from march.datasets.c4 import VOCAB_SIZE

from march.models.utils import LayerNorm, TransformerComponentBase


NL = TensorType["N", "L"]
NLD = TensorType["N", "L", "D"]
NLL = TensorType["N", "L", "L"]
N1LL = TensorType["N", "1", "L_q", "L_k"]
NLHDkv = TensorType["N", "L", "H", "Dkv"]
NHLDkv = TensorType["N", "H", "L", "Dkv"]


@dataclass
class BaselineV2Config(PretrainedConfig):
    dim_model: int = 896
    num_decoder_layers: int = 8
    dim_qkv: int = 64

    feedforward_scale: float = (2 / 3) * 4

    vocab_size: int = VOCAB_SIZE
    dtype = bfloat16 if is_torch_bf16_gpu_available() else float32

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


class BaselineV2Attention(TransformerComponentBase):
    def __init__(self, config: BaselineV2Config) -> None:
        super().__init__(config)

        D, HDkv = config.dim_model, config.num_heads * config.dim_qkv
        self.w_q = Linear(config, D, HDkv)
        self.w_k = Linear(config, D, HDkv)
        self.w_v = Linear(config, D, HDkv)
        self.w_o = Linear(config, HDkv, D)
        self.rotary_emb = RotaryEmbedding(config.dim_qkv)

    def forward(
        self, embeds: NLD, attention_mask: N1LL, encoder_embeds: Optional[NLD] = None
    ) -> NLD:
        config = self.config
        N, L_q, D = embeds.size()
        L_k = encoder_embeds.size(1) if encoder_embeds is not None else L_q
        H, Dkv = config.num_heads, config.dim_qkv

        key_value_embeds = encoder_embeds if encoder_embeds is not None else embeds
        query: NHLDkv = self.w_q(embeds).view(N, L_q, H, Dkv).transpose(1, 2)
        key: NHLDkv = self.w_k(key_value_embeds).view(N, L_k, H, Dkv).transpose(1, 2)
        query, key = self.rotary_emb(query, key)
        query: NLHDkv = query.transpose(1, 2)
        key: NLHDkv = key.transpose(1, 2)
        value: NLHDkv = self.w_v(key_value_embeds).view(N, L_k, H, Dkv)

        attention_values: NLHDkv = memory_efficient_attention(query, key, value, attention_mask)

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

        try:
            from apex.normalization import FusedRMSNorm

            ALL_LAYERNORM_LAYERS.append(FusedRMSNorm)
            layernorm_fn = lambda: FusedRMSNorm(config.dim_model, eps=1e-6)
        except ImportError:
            layernorm_fn = lambda: LayerNorm(config)

        self.layers = ModuleList()
        for _ in range(config.num_decoder_layers):
            self.layers.append(layer := Module())

            layer.selfattn_ln, layer.selfattn = layernorm_fn(), attn_cls(config)

            if is_decoder:
                layer.crossattn_ln, layer.crossattn = layernorm_fn(), attn_cls(config)

            layer.ff_ln, layer.ff = layernorm_fn(), ff_cls(config)

        self.final_layernorm = layernorm_fn()

    def forward(
        self,
        embeds: NLD,
        mask: N1LL,
        encoder_embeds: Optional[NLD] = None,
        encoder_mask: Optional[N1LL] = None,
    ) -> NLD:
        for layer in self.layers:
            embeds = embeds + layer.selfattn(layer.selfattn_ln(embeds), mask)

            if self.is_decoder:
                embeds = embeds + layer.crossattn(
                    layer.crossattn_ln(embeds), encoder_mask, encoder_embeds
                )

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

    def forward(
        self,
        input_ids: NL,  # encoder input ids
        attention_mask: NL,  # encoder attention mask
        decoder_input_ids: NL,
        decoder_attention_mask: NLL,
        labels: NL,
    ) -> Seq2SeqLMOutput:
        config = self.config

        def convert_attention_mask(mask: Union[NL, NLL]) -> N1LL:
            N, L_k = mask.size(0), mask.size(-1)
            if mask.dim() == 2:
                L_q = L_k
                mask = mask.view(N, 1, 1, L_k).repeat(1, 1, L_q, 1)
            else:
                L_q = mask.size(1)
                mask = mask.view(N, 1, L_q, L_k)

            return mask.to(config.dtype) * finfo(config.dtype).min

        encoder_mask: N1LL = convert_attention_mask(attention_mask)
        encoder_embeds: NLD = self.encoder(
            embedding(self.embedding.weight, input_ids), encoder_mask
        )

        decoder_mask: N1LL = convert_attention_mask(decoder_attention_mask)
        decoder_embeds: NLD = embedding(self.embedding.weight, decoder_input_ids)
        decoder_embeds: NLD = self.decoder(
            decoder_embeds, decoder_mask, encoder_embeds, encoder_mask
        )

        logits: NLD = self.embedding(decoder_embeds * (config.dim_model**-0.5))

        loss = cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(loss=loss, logits=logits)
