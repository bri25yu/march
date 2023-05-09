from march.models.baseline import *
from march.models.utils import *

from torch.nn import Parameter, Linear
from torch import zeros


__all__ = ["TPEmbeddingsTransformer", "TPEmbeddingsConfig"]


# Reimplementing the Tensor Product Transformer with discrete role embeddings
# for roles (relations), and output of transformer attention as the fillers.
# https://arxiv.org/abs/2106.01317


@dataclass
class TPEmbeddingsConfig(TransformerConfig):
    num_roles: int = 100


class TPEmbeddingsAttention(TransformerComponentBase):
    
    def __init__(self, config: TPEmbeddingsConfig) -> None:
        super().__init__(config)

        # Create role embeddings
        self.role_embeddings = Parameter(zeros(config.num_roles, config.dim_qkv))
        
        # Create r linear layer
        self.w_r = Linear(config.dim_model, config.num_roles * config.num_heads, bias=False)

    def init_weights(self) -> None:
        # Init to mean 0 variance 1, it is recombined later as the role embeddings by multiplying
        # by the attention output. Normalized later also so it doesn't matter too much.
        self.role_embeddings.data.normal_(mean=0.0, std=1.0)

        # Init to mean 0 variance 1 / dim_model ** -0.5, we are summing over dim_model in the weight
        # multiplication, so we need to scale by 1 / dim_model ** -0.5
        self.w_r.weight.data.normal_(mean=0.0, std=(self.config.dim_model) ** -0.5)

    def forward(
        self,
        input_embeds: SequenceInputEmbeds,
    ) -> AttentionOutput:
        # input_embeds is the output of normal attention
        # (N, L, D)

        # normalize role embeddings
        normalized_role_embeddings = self.role_embeddings / self.role_embeddings.norm(dim=1, keepdim=True)

        # Compute role attention
        role_attention = self.w_r(input_embeds)
        # (N, L, (N_R * H))

        # Reshape to head sensitive for roles
        role_attention_heads = self.reshape_to_head_sensitive_roles(role_attention)
        # (N, H, L, N_R)

        # softmax over roles dimension
        selected_role_indices_heads = role_attention_heads.softmax(dim=-1)
        # (N, H, L, N_R)

        # then cast to role embeddings
        selected_role_heads: MultiHeadedEmbeds = matmul(selected_role_indices_heads, normalized_role_embeddings)
        # (N, H, L, D_kv)

        # Do concatenation of role embeddings 
        selected_roles: SequenceInputEmbeds = AttentionBase.reshape_to_head_insensitive(selected_role_heads)
        # (N, L, H * D_kv) == (N, L, D)

        # Compute tensor product representation with hadamard product of roles and fillers
        # Elementwise product over the last D dimension doesn't change the dimension.
        tp_attention_output: SequenceInputEmbeds = selected_roles * input_embeds
        # (N, L, D)

        return tp_attention_output


    def reshape_to_head_sensitive_roles(self, input_embeds: SequenceInputEmbeds) -> MultiHeadedEmbeds:
        # input_embeds shape (N, L, (N_R * H)) -> (N, H, L, N_R)
        config = self.config

        batch_size, sequence_length = input_embeds.size()[0], input_embeds.size()[1]

        input_embeds = input_embeds.reshape(batch_size, sequence_length, config.num_heads, config.num_roles)
        # Permute to shape (N, H, L, N_R)
        input_embeds = input_embeds.permute(0, 2, 1, 3)

        return input_embeds


class TPEmbeddingsEncoder(EncoderBase):
    ATTENTION_CLS = BaselineAttention
    FEEDFORWARD_CLS = BaselineFeedforward
    ROLE_ATTENTION_CLS = TPEmbeddingsAttention
    
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        # Init same as encoder base, but add role attention layers
        self.role_attention_layers: List[AttentionBase] = ModuleList(
            [self.ROLE_ATTENTION_CLS(config) for _ in range(config.num_layers)]
        )

    def forward(self, input_embeds: SequenceInputEmbeds, attention_mask: SequenceInputIds) -> AttentionOutput:
        config: TransformerConfig = self.config

        encoder_key_value_states: List[KeyValueStates] = []
        for i in range(config.num_layers // 2):
            self_attention_layernorm: LayerNorm = self.layernorms[2 * i]
            self_attention_layer: AttentionBase = self.self_attention_layers[i]

            # Select role attention layer
            role_attention_layer: AttentionBase = self.role_attention_layers[i]

            feedforward_layernorm: LayerNorm = self.layernorms[2 * i + 1]
            feedforward_layer: TransformerComponentBase = self.feedforward_layers[i]

            normed_input_embeds: SequenceInputEmbeds = self_attention_layernorm(input_embeds)
            self_attention_output: AttentionOutput = self_attention_layer(normed_input_embeds, attention_mask)

            # Add role embedding + binding operation here:
            role_binded_output: SequenceInputEmbeds = role_attention_layer(self_attention_output.input_embeds)

            input_embeds: SequenceInputEmbeds = input_embeds + dropout(role_binded_output, p=config.dropout_prob, training=self.training)
            encoder_key_value_states.append(self_attention_output.key_value_states)

            normed_input_embeds: SequenceInputEmbeds = feedforward_layernorm(input_embeds)
            feedforward_output: SequenceInputEmbeds = feedforward_layer(normed_input_embeds)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(feedforward_output, p=config.dropout_prob, training=self.training)

        input_embeds: SequenceInputEmbeds = self.layernorms[-1](input_embeds)

        return AttentionOutput(input_embeds=input_embeds, key_value_states=encoder_key_value_states)


class TPEmbeddingsDecoder(DecoderBase):
    ATTENTION_CLS = BaselineAttention
    FEEDFORWARD_CLS = BaselineFeedforward
    ROLE_ATTENTION_CLS = TPEmbeddingsAttention
    
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        # Init same as decoder base, but add role attention layers
        self.role_attention_layers: List[AttentionBase] = ModuleList(
            [self.ROLE_ATTENTION_CLS(config) for _ in range(config.num_layers)]
        )

    def forward(
        self,
        input_embeds: SequenceInputEmbeds,
        attention_mask: SequenceInputIds,
        encoder_key_value_states: List[KeyValueStates],
        encoder_attention_mask: SequenceInputIds,
    ) -> AttentionOutput:
        config: TransformerConfig = self.config

        for i in range(config.num_layers // 2):
            self_attention_layernorm: LayerNorm = self.layernorms[2 * i]
            self_attention_layer: AttentionBase = self.self_attention_layers[i]

            # Select role attention layer
            role_attention_layer: AttentionBase = self.role_attention_layers[i]

            cross_attention_layernorm: LayerNorm = self.layernorms[2 * i + 1]
            cross_attention_key_value_states = encoder_key_value_states[i]
            cross_attention_layer: AttentionBase = self.cross_attention_layers[i]
            feedforward_layernorm: LayerNorm = self.layernorms[2 * i + 2]
            feedforward_layer: TransformerComponentBase = self.feedforward_layers[i]

            normed_input_embeds: SequenceInputEmbeds = self_attention_layernorm(input_embeds)
            self_attention_output: AttentionOutput = self_attention_layer(normed_input_embeds, attention_mask)

            # Add role embedding + binding operation here:
            role_binded_output: SequenceInputEmbeds = role_attention_layer(self_attention_output.input_embeds)

            input_embeds: SequenceInputEmbeds = input_embeds + dropout(role_binded_output, p=config.dropout_prob, training=self.training)

            normed_input_embeds: SequenceInputEmbeds = cross_attention_layernorm(input_embeds)
            cross_attention_output: AttentionOutput = cross_attention_layer(normed_input_embeds, encoder_attention_mask, cross_attention_key_value_states)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(cross_attention_output.input_embeds, p=config.dropout_prob, training=self.training)

            normed_input_embeds: SequenceInputEmbeds = feedforward_layernorm(input_embeds)
            feedforward_output: SequenceInputEmbeds = feedforward_layer(normed_input_embeds)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(feedforward_output, p=config.dropout_prob, training=self.training)

        input_embeds: SequenceInputEmbeds = self.layernorms[-1](input_embeds)

        return AttentionOutput(input_embeds=input_embeds, key_value_states=None)


class TPEmbeddingsTransformer(BaselineTransformer):
    ENCODER_CLS = TPEmbeddingsEncoder
    DECODER_CLS = TPEmbeddingsDecoder
