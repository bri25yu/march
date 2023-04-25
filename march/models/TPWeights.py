from march.models.baseline import *
from march.models.utils import *


__all__ = ["TPWeightsTransformer"]


# Reimplementing the Tensor Product Transformer with weight matrix w_r
# for roles (relations), and output of transformer attention as the fillers.
# https://arxiv.org/abs/1910.06611


class TPWeightsAttention(AttentionBase):
    def __init__(self, config: TransformerConfig, is_cross_attention: bool) -> None:
        super().__init__(config, is_cross_attention)

        self.w_q = Linear(config.dim_model, config.num_heads * config.dim_qkv, bias=False)
        self.w_k = Linear(config.dim_model, config.num_heads * config.dim_qkv, bias=False)
        self.w_v = Linear(config.dim_model, config.num_heads * config.dim_qkv, bias=False)
        self.w_o = Linear(config.num_heads * config.dim_qkv, config.dim_model, bias=False)

        # Add w_r relation weight matrix, with same dimension as dim_qkv
        self.w_r = Linear(config.dim_model, config.num_heads * config.dim_qkv, bias=False)

    def init_weights(self) -> None:
        config = self.config

        self.w_q.weight.data.normal_(mean=0.0, std=(config.dim_model * config.dim_qkv) ** -0.5)
        self.w_k.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)
        self.w_v.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)
        self.w_o.weight.data.normal_(mean=0.0, std=(config.num_heads * config.dim_qkv) ** -0.5)

        # Init w_r relation weight matrix with mean 0 and std 1/sqrt(dim_dkv)
        # Point of difference here: we use dim_model instead of dim_qkv, which should correctly
        # produce matrix R = X @ W_r with mean 0 variance 1
        # https://github.com/ischlag/TP-Transformer/blob/master/models/tp-transformer.py#L260
        self.w_r.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)

    def forward(
        self,
        input_embeds: SequenceInputEmbeds,
        attention_mask: SequenceInputIds=None,
        encoder_key_value_states: KeyValueStates=None,
    ) -> AttentionOutput:
        config = self.config

        if not self.is_cross_attention:
            # Project input_embeds to query, key, value, and relation
            attention_values: List[SequenceInputEmbeds] = self.w_q(input_embeds), self.w_k(input_embeds), self.w_v(input_embeds), self.w_r(input_embeds)
        else:
            key, value = encoder_key_value_states

            query: SequenceInputEmbeds = self.w_q(input_embeds)
            # Use input_embeds instead of encoder outputs for relation
            relation: SequenceInputEmbeds = self.w_r(input_embeds)
            key: SequenceInputEmbeds = self.w_k(self.reshape_to_head_insensitive(key))
            value: SequenceInputEmbeds = self.w_v(self.reshape_to_head_insensitive(value))

            attention_values: List[SequenceInputEmbeds] = (query, key, value, relation)

        query, key, value, relation = list(map(self.reshape_to_head_sensitive, attention_values))

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

        # Tensor Product Attention: Apply roles to fillers using hadamard product
        attention_values: MultiHeadedEmbeds = attention_values * relation

        attention_values: SequenceInputEmbeds = self.reshape_to_head_insensitive(attention_values)

        attention_output: SequenceInputEmbeds = self.w_o(attention_values)

        return AttentionOutput(attention_output, (key, value))


class TPWeightsEncoder(EncoderBase):
    ATTENTION_CLS = TPWeightsAttention
    FEEDFORWARD_CLS = BaselineFeedforward


class TPWeightsDecoder(DecoderBase):
    ATTENTION_CLS = TPWeightsAttention
    FEEDFORWARD_CLS = BaselineFeedforward


class TPWeightsTransformer(BaselineTransformer):
    ENCODER_CLS = TPWeightsEncoder
    DECODER_CLS = TPWeightsDecoder
