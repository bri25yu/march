from march.models.baseline import *
from march.models.utils import *

from torch.nn import Parameter, Linear
from torch import zeros
from enum import Enum


__all__ = ["TPEmbeddingsBeforeW_oEncDecSharedTransformer",
    "TPEmbeddingsBeforeW_oEncDecNotSharedTransformer",
    "TPEmbeddingsBeforeW_oConfig"]


# TPEmbeddings but with the role embeddings applied 
# directly to the attention output instead of after the w_o linear layer.

# class RoleDecoderType(Enum):
#     SELF_ATTENTION = 0
#     CROSS_ATTENTION = 1
#     BOTH = 2


@dataclass
class TPEmbeddingsBeforeW_oConfig(TransformerConfig):
    num_roles: int = 100
    role_attention_type: int = 0


class TPEmbeddingsBeforeW_oAttentionComponent(TransformerComponentBase):
    
    def __init__(self, config: TPEmbeddingsBeforeW_oConfig) -> None:
        super().__init__(config)

        # Create role embeddings
        self.role_embeddings = Parameter(zeros(config.num_roles, config.dim_qkv))
        
        # Create r linear layer
        self.w_r = Linear(config.dim_qkv, config.num_roles, bias=False)

    def init_weights(self) -> None:
        # Init to mean 0 variance 1, it is recombined later as the role embeddings by multiplying
        # by the attention output. Normalized later also so it doesn't matter too much.
        self.role_embeddings.data.normal_(mean=0.0, std=1.0)

        # Init to mean 0 variance 1 / dim_model ** -0.5, we are summing over dim_model in the weight
        # multiplication, so we need to scale by 1 / dim_model ** -0.5
        self.w_r.weight.data.normal_(mean=0.0, std=(self.config.dim_model) ** -0.5)

    def forward(
        self,
        input_embeds_heads: MultiHeadedEmbeds,
    ) -> MultiHeadedEmbeds:
        # input_embeds is the output of normal attention before recombination to heads
        # (N, H, L, D_kv)

        # normalize role embeddings
        normalized_role_embeddings = self.role_embeddings / self.role_embeddings.norm(dim=1, keepdim=True)

        # Compute role attention
        role_attention_heads = self.w_r(input_embeds_heads)
        # (N, H, L, N_R)

        # softmax over roles dimension
        selected_role_indices_heads = role_attention_heads.softmax(dim=-1)
        # (N, H, L, N_R)

        # then cast to role embeddings
        selected_role_heads: MultiHeadedEmbeds = matmul(selected_role_indices_heads, normalized_role_embeddings)
        # (N, H, L, D_kv)

        # Compute tensor product representation with hadamard product of roles and fillers
        # Elementwise product over the last D dimension doesn't change the dimension.
        tp_attention_output_heads: SequenceInputEmbeds = selected_role_heads * input_embeds_heads
        # (N, H, L, D_kv)

        return tp_attention_output_heads


class TPEmbeddingsBeforeW_oAttention(BaselineAttention):
    def __init__(self, config: TPEmbeddingsBeforeW_oConfig, is_cross_attention: bool, shared_role_embeddings: TPEmbeddingsBeforeW_oAttentionComponent) -> None:
        super().__init__(config, is_cross_attention)
        if shared_role_embeddings is None:
            self.role_embeddings = TPEmbeddingsBeforeW_oAttentionComponent(config)
        else:
            # Not shared, init for this attention layer
            self.role_embeddings = shared_role_embeddings


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

        # Before concat and w_o, apply role embeddings
        attention_values: MultiHeadedEmbeds = self.role_embeddings.forward(attention_values)
        # Output (N, H, L, D_kv)

        attention_values: SequenceInputEmbeds = self.reshape_to_head_insensitive(attention_values)

        attention_output: SequenceInputEmbeds = self.w_o(attention_values)

        return AttentionOutput(attention_output, (key, value))



class TPEmbeddingsBeforeW_oEncoder(EncoderBase):
    ATTENTION_CLS = TPEmbeddingsBeforeW_oAttention
    FEEDFORWARD_CLS = BaselineFeedforward
    
    def __init__(self, config: TransformerConfig, shared_role_embeddings: TPEmbeddingsBeforeW_oAttentionComponent) -> None:
        TransformerComponentBase.__init__(self, config)
        self.layernorms = ModuleList([LayerNorm(config) for _ in range(config.num_layers + 1)])
        
        # Init same as encoder base, but share role attention layers if applicable
        self.self_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(config, is_cross_attention=False, shared_role_embeddings=shared_role_embeddings) for _ in range(config.num_layers // 2)]
        )

        self.feedforward_layers: List[TransformerComponentBase] = ModuleList(
            [self.FEEDFORWARD_CLS(config) for _ in range(config.num_layers // 2)]
        )

    # Normal init without shared_role_embeddings will create a new one for each layer since that is the default for the 
    # individual attention layers


class TPEmbeddingsBeforeW_oDecoder(DecoderBase):
    ATTENTION_CLS = BaselineAttention
    ROLE_ATTENTION_CLS = TPEmbeddingsBeforeW_oAttention
    FEEDFORWARD_CLS = BaselineFeedforward
    
    def __init__(self, config: TPEmbeddingsBeforeW_oConfig, shared_role_embeddings: TPEmbeddingsBeforeW_oAttentionComponent) -> None:
        TransformerComponentBase.__init__(self, config)

        self.layernorms = ModuleList([LayerNorm(config) for _ in range(((config.num_layers // 2) * 3) + 1)])

        # Init same as decoder base, but share role attention layers if applicable based on the enum
        match config.role_attention_type:
            case 0: # RoleDecoderType.SELF_ATTENTION
                self.self_attention_layers: List[AttentionBase] = ModuleList(
                    [self.ROLE_ATTENTION_CLS(config, is_cross_attention=False, shared_role_embeddings=shared_role_embeddings) for _ in range(config.num_layers // 2)]
                )
                self.cross_attention_layers: List[AttentionBase] = ModuleList(
                    [self.ATTENTION_CLS(config, is_cross_attention=True) for _ in range(config.num_layers // 2)]
                )
            case 1: # RoleDecoderType.CROSS_ATTENTION
                self.self_attention_layers: List[AttentionBase] = ModuleList(
                    [self.ATTENTION_CLS(config, is_cross_attention=False) for _ in range(config.num_layers // 2)]
                )
                self.cross_attention_layers: List[AttentionBase] = ModuleList(
                    [self.ROLE_ATTENTION_CLS(config, is_cross_attention=True, shared_role_embeddings=shared_role_embeddings) for _ in range(config.num_layers // 2)]
                )
            case 2: # RoleDecoderType.BOTH
                self.self_attention_layers: List[AttentionBase] = ModuleList(
                    [self.ROLE_ATTENTION_CLS(config, is_cross_attention=False, shared_role_embeddings=shared_role_embeddings) for _ in range(config.num_layers // 2)]
                )
                self.cross_attention_layers: List[AttentionBase] = ModuleList(
                    [self.ROLE_ATTENTION_CLS(config, is_cross_attention=True, shared_role_embeddings=shared_role_embeddings) for _ in range(config.num_layers // 2)]
                )
            case _:
                raise NotImplementedError(f"Role attention type {config.role_attention_type} not implemented")

        self.feedforward_layers: List[TransformerComponentBase] = ModuleList(
            [self.FEEDFORWARD_CLS(config) for _ in range(config.num_layers // 2)]
        )
    
    # Normal init without shared_role_embeddings will just do a normal decoder since it is not using the 
    # role_attention_cls by default


class TPEmbeddingsBeforeW_oEncDecSharedTransformer(BaselineTransformer):
    ENCODER_CLS = TPEmbeddingsBeforeW_oEncoder
    DECODER_CLS = TPEmbeddingsBeforeW_oDecoder

    def __init__(self, config: TPEmbeddingsBeforeW_oConfig) -> None:
        TransformerComponentBase.__init__(self, config)

        self.config = config

        self.embedding: TensorType["D", "V"] = Linear(config.dim_model, config.vocab_size, bias=False)
        self.position_encoding = self.POSITION_ENCODING_CLS(config)

        self.shared_role_embeddings = TPEmbeddingsBeforeW_oAttentionComponent(config)

        self.encoder = self.ENCODER_CLS(config, self.shared_role_embeddings)
        self.decoder = self.DECODER_CLS(config, self.shared_role_embeddings)


class TPEmbeddingsBeforeW_oEncDecNotSharedTransformer(BaselineTransformer):
    ENCODER_CLS = TPEmbeddingsBeforeW_oEncoder
    DECODER_CLS = TPEmbeddingsBeforeW_oDecoder

    def __init__(self, config: TPEmbeddingsBeforeW_oConfig) -> None:
        TransformerComponentBase.__init__(self, config)

        self.config = config

        self.embedding: TensorType["D", "V"] = Linear(config.dim_model, config.vocab_size, bias=False)
        self.position_encoding = self.POSITION_ENCODING_CLS(config)

        self.enc_shared_role_embeddings = TPEmbeddingsBeforeW_oAttentionComponent(config)
        self.dec_shared_role_embeddings = TPEmbeddingsBeforeW_oAttentionComponent(config)

        self.encoder = self.ENCODER_CLS(config, self.enc_shared_role_embeddings)
        self.decoder = self.DECODER_CLS(config, self.dec_shared_role_embeddings)
