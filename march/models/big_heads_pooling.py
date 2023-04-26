from abc import abstractmethod

from march.models.baseline import *
from march.models.utils import *


__all__ = ["BigHeadsPoolingTransformer", "BigHeadsPoolingTransformerConfig"]

# Goes from D_model to H * D_model, then each head-wise attention is added together
# before being passed into w_o. So D_kv := D_model
@dataclass
class BigHeadsPoolingTransformerConfig(TransformerConfig):
    dim_qkv: Union[None, int] = None

    num_heads: int = 12

    def __post_init__(self) -> None:
        self.dim_qkv = self.dim_model

        if self.dim_feedforward is None:
            self.dim_feedforward = self.dim_model * self.feedforward_scale


class BigHeadsPoolingAttention(BaselineAttention):
    def __init__(self, config: BigHeadsPoolingTransformerConfig, is_cross_attention: bool) -> None:
        super().__init__(config, is_cross_attention)

        self.w_q = Linear(config.dim_model, config.num_heads * config.dim_qkv, bias=False)

        # START big heads attention pooling
        # Make w_o a linear layer from D_model to D_model 
        # self.w_o = Linear(config.num_heads * config.dim_qkv, config.dim_model, bias=False)
        self.w_o = Linear(config.dim_model, config.dim_model, bias=False)
        
        # END big heads attention

        if self.is_cross_attention:
            self.w_k = Linear(config.num_heads * config.dim_qkv, config.num_heads * config.dim_qkv, bias=False)
            self.w_v = Linear(config.num_heads * config.dim_qkv, config.num_heads * config.dim_qkv, bias=False)
        else:
            self.w_k = Linear(config.dim_model, config.num_heads * config.dim_qkv, bias=False)
            self.w_v = Linear(config.dim_model, config.num_heads * config.dim_qkv, bias=False)


    def init_weights(self) -> None:
        config = self.config

        self.w_q.weight.data.normal_(mean=0.0, std=(config.dim_model * config.dim_qkv) ** -0.5)
        self.w_o.weight.data.normal_(mean=0.0, std=(config.num_heads * config.dim_qkv) ** -0.5)

        # TODO change w_v init

        if self.is_cross_attention:
            self.w_k.weight.data.normal_(mean=0.0, std=(config.num_heads * config.dim_qkv) ** -0.5)
            self.w_v.weight.data.normal_(mean=0.0, std=(config.num_heads * config.dim_qkv) ** -0.5)
        else:
            self.w_k.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)
            self.w_v.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)

    # Do pooling here instead of shaping to head insensitive back again.
    def forward(
        self,
        input_embeds: SequenceInputEmbeds,
        attention_mask: SequenceInputIds=None,
        encoder_key_value_states: KeyValueStates=None,
    ) -> AttentionOutput:
        config = self.config

        if not self.is_cross_attention:
            attention_values: List[SequenceInputEmbeds] = self.w_q(input_embeds), self.w_k(input_embeds), self.w_v(input_embeds)
        else:
            key, value = encoder_key_value_states

            query: SequenceInputEmbeds = self.w_q(input_embeds)
            key: SequenceInputEmbeds = self.w_k(self.reshape_to_head_insensitive(key))
            value: SequenceInputEmbeds = self.w_v(self.reshape_to_head_insensitive(value))

            attention_values: List[SequenceInputEmbeds] = (query, key, value)

        query, key, value = list(map(self.reshape_to_head_sensitive, attention_values))

        attention_logits: MultiHeadedAttention = matmul(query, key.transpose(2, 3))

        if attention_mask is not None:
            if len(attention_mask.size()) == 2:
                query_length = 1
                batch_size, key_length = attention_mask.size()
            elif len(attention_mask.size()) == 3:
                batch_size, query_length, key_length = attention_mask.size()

            attention_mask = attention_mask.reshape(batch_size, 1, query_length, key_length)
            attention_mask = attention_mask.to(attention_logits.dtype) * finfo(attention_logits.dtype).min
            attention_logits: MultiHeadedAttention = attention_logits + attention_mask

        attention_probs: MultiHeadedAttention = attention_logits.softmax(dim=3)
        attention_probs: MultiHeadedAttention = dropout(attention_probs, p=config.dropout_prob, training=self.training)

        attention_values: MultiHeadedEmbeds = matmul(attention_probs, value)

        # Here do pooling 
        # attention_values: SequenceInputEmbeds = self.reshape_to_head_insensitive(attention_values)
        # permute to (N, L, H, D_kv)
        input_embeds = input_embeds.permute(0, 2, 1, 3)
        # Sum over H dimension
        attention_values: SequenceInputEmbeds = attention_values.sum(dim=2)

        attention_output: SequenceInputEmbeds = self.w_o(attention_values)

        return AttentionOutput(attention_output, (key, value))


class BigHeadsPoolingEncoder(EncoderBase):
    ATTENTION_CLS = BigHeadsPoolingAttention
    FEEDFORWARD_CLS = BaselineFeedforward


class BigHeadsPoolingDecoder(DecoderBase):
    ATTENTION_CLS = BigHeadsPoolingAttention
    FEEDFORWARD_CLS = BaselineFeedforward


class BigHeadsPoolingTransformer(BaselineTransformer):
    ENCODER_CLS = BigHeadsPoolingEncoder
    DECODER_CLS = BigHeadsPoolingDecoder
