from typing import List, Type
from torchtyping import TensorType

from abc import abstractmethod

from torch import finfo, long, matmul, ones, triu
from torch.nn import CrossEntropyLoss, Embedding, Linear, ModuleList
from torch.nn.functional import dropout, embedding, relu

from transformers.modeling_outputs import Seq2SeqLMOutput

from march.models.utils import *


__all__ = [
    "CrossEntropyLoss",
    "FloatTensor",
    "Linear",
    "ModuleList",
    "dropout",
    "embedding",
    "finfo",
    "matmul",
    "Seq2SeqLMOutput",
    "EncoderBase",
    "DecoderBase",
    "TransformerBase",
    "BaselineAttention",
    "BaselineFeedforward",
    "BaselineEncoder",
    "BaselineDecoder",
    "BaselineTransformer",
]


class EncoderBase(TransformerComponentBase):
    @property
    @abstractmethod
    def ATTENTION_CLS(self) -> Type[AttentionBase]:
        pass

    @property
    @abstractmethod
    def FEEDFORWARD_CLS(self) -> Type[TransformerComponentBase]:
        pass

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)

        self.layernorms = ModuleList([LayerNorm(config) for _ in range(config.num_layers + 1)])
        self.self_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(config, is_cross_attention=False, has_relative_attention_bias=i==0) for i in range(config.num_layers // 2)]
        )
        self.feedforward_layers: List[TransformerComponentBase] = ModuleList(
            [self.FEEDFORWARD_CLS(config) for _ in range(config.num_layers // 2)]
        )

    def forward(self, input_embeds: SequenceInputEmbeds, attention_mask: SequenceInputIds) -> AttentionOutput:
        config: TransformerConfig = self.config

        encoder_key_value_states: List[KeyValueStates] = []
        for i in range(config.num_layers // 2):
            self_attention_layernorm: LayerNorm = self.layernorms[2 * i]
            self_attention_layer: AttentionBase = self.self_attention_layers[i]
            feedforward_layernorm: LayerNorm = self.layernorms[2 * i + 1]
            feedforward_layer: TransformerComponentBase = self.feedforward_layers[i]

            normed_input_embeds: SequenceInputEmbeds = self_attention_layernorm(input_embeds)
            self_attention_output: AttentionOutput = self_attention_layer(normed_input_embeds, attention_mask)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(self_attention_output.input_embeds, p=config.dropout_prob, training=self.training)
            encoder_key_value_states.append(self_attention_output.key_value_states)

            normed_input_embeds: SequenceInputEmbeds = feedforward_layernorm(input_embeds)
            feedforward_output: SequenceInputEmbeds = feedforward_layer(normed_input_embeds)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(feedforward_output, p=config.dropout_prob, training=self.training)

        input_embeds: SequenceInputEmbeds = self.layernorms[-1](input_embeds)

        return AttentionOutput(input_embeds=input_embeds, key_value_states=encoder_key_value_states)


class DecoderBase(TransformerComponentBase):
    @property
    @abstractmethod
    def ATTENTION_CLS(self) -> Type[AttentionBase]:
        pass

    @property
    @abstractmethod
    def FEEDFORWARD_CLS(self) -> Type[TransformerComponentBase]:
        pass

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)

        self.layernorms = ModuleList([LayerNorm(config) for _ in range(((config.num_layers // 2) * 3) + 1)])
        self.self_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(config, is_cross_attention=False, has_relative_attention_bias=i==0) for i in range(config.num_layers // 2)]
        )
        self.cross_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(config, is_cross_attention=True) for _ in range(config.num_layers // 2)]
        )
        self.feedforward_layers: List[TransformerComponentBase] = ModuleList(
            [self.FEEDFORWARD_CLS(config) for _ in range(config.num_layers // 2)]
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
            cross_attention_layernorm: LayerNorm = self.layernorms[2 * i + 1]
            cross_attention_key_value_states = encoder_key_value_states[i]
            cross_attention_layer: AttentionBase = self.cross_attention_layers[i]
            feedforward_layernorm: LayerNorm = self.layernorms[2 * i + 2]
            feedforward_layer: TransformerComponentBase = self.feedforward_layers[i]

            normed_input_embeds: SequenceInputEmbeds = self_attention_layernorm(input_embeds)
            self_attention_output: AttentionOutput = self_attention_layer(normed_input_embeds, attention_mask)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(self_attention_output.input_embeds, p=config.dropout_prob, training=self.training)

            normed_input_embeds: SequenceInputEmbeds = cross_attention_layernorm(input_embeds)
            cross_attention_output: AttentionOutput = cross_attention_layer(normed_input_embeds, encoder_attention_mask, cross_attention_key_value_states)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(cross_attention_output.input_embeds, p=config.dropout_prob, training=self.training)

            normed_input_embeds: SequenceInputEmbeds = feedforward_layernorm(input_embeds)
            feedforward_output: SequenceInputEmbeds = feedforward_layer(normed_input_embeds)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(feedforward_output, p=config.dropout_prob, training=self.training)

        input_embeds: SequenceInputEmbeds = self.layernorms[-1](input_embeds)

        return AttentionOutput(input_embeds=input_embeds, key_value_states=None)


class TransformerBase(TransformerComponentBase):
    @property
    @abstractmethod
    def ENCODER_CLS(self) -> Type[EncoderBase]:
        pass

    @property
    @abstractmethod
    def DECODER_CLS(self) -> Type[DecoderBase]:
        pass

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)

        self.config = config

        self.embedding: TensorType["D", "V"] = Linear(config.dim_model, config.vocab_size, bias=False)

        self.encoder = self.ENCODER_CLS(config)
        self.decoder = self.DECODER_CLS(config)

    def init_weights(self) -> None:
        self.embedding.weight.data.normal_(mean=0.0, std=1.0)

    def forward(
        self,
        input_ids: SequenceInputIds,
        attention_mask: SequenceInputIds,
        decoder_input_ids: SequenceInputIds,
        labels: SequenceInputIds,
    ) -> Seq2SeqLMOutput:
        config = self.config

        input_embeds: SequenceInputEmbeds = embedding(input_ids, self.embedding.weight)
        encoder_outputs: AttentionOutput = self.encoder(input_embeds, attention_mask)

        decoder_input_embeds: SequenceInputEmbeds = embedding(decoder_input_ids, self.embedding.weight)
        decoder_attention_mask = self.create_decoder_attention_mask(decoder_input_ids)
        decoder_outputs: AttentionOutput = self.decoder(
            decoder_input_embeds, decoder_attention_mask, encoder_outputs.key_value_states, attention_mask,
        )

        sequence_output = decoder_outputs.input_embeds
        sequence_output = sequence_output * (config.dim_model ** -0.5)

        lm_logits = self.embedding(sequence_output)

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(loss=loss, logits=lm_logits)

    def create_decoder_attention_mask(self, decoder_input_ids: SequenceInputIds) -> TensorType["N", "L_out", "L_out"]:
        """
        1 for should mask, 0 otherwise.
        """
        batch_size, sequence_length = decoder_input_ids.size()

        no_loss_mask: TensorType["N", 1, "L_out"] = decoder_input_ids[:, None, :] == -100
        causal_mask: TensorType["L_out", "L_out"] = triu(ones(sequence_length, sequence_length, dtype=long), diagonal=1)[None, :, :].to(device=decoder_input_ids.device)
        decoder_attention_mask: TensorType["N", "L_out", "L_out"] = no_loss_mask | causal_mask
        assert decoder_attention_mask.size() == (batch_size, sequence_length, sequence_length), f"Expected decoder attention mask of shape {(batch_size, sequence_length, sequence_length)}, but got {decoder_attention_mask.size()}."

        return decoder_attention_mask

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineAttention(AttentionBase):
    def __init__(self, config: TransformerConfig, is_cross_attention: bool, has_relative_attention_bias: bool=False) -> None:
        super().__init__(config, is_cross_attention)

        self.has_relative_attention_bias = has_relative_attention_bias

        self.w_q = Linear(config.dim_model, config.num_heads * config.dim_qkv, bias=False)
        self.w_o = Linear(config.num_heads * config.dim_qkv, config.dim_model, bias=False)

        if self.is_cross_attention:
            self.w_k = Linear(config.num_heads * config.dim_qkv, config.num_heads * config.dim_qkv, bias=False)
            self.w_v = Linear(config.num_heads * config.dim_qkv, config.num_heads * config.dim_qkv, bias=False)
        else:
            self.w_k = Linear(config.dim_model, config.num_heads * config.dim_qkv, bias=False)
            self.w_v = Linear(config.dim_model, config.num_heads * config.dim_qkv, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = Embedding(config.relative_attention_num_buckets, config.num_heads)

    def init_weights(self) -> None:
        config = self.config

        self.w_q.weight.data.normal_(mean=0.0, std=(config.dim_model * config.dim_qkv) ** -0.5)
        self.w_o.weight.data.normal_(mean=0.0, std=(config.num_heads * config.dim_qkv) ** -0.5)

        if self.is_cross_attention:
            self.w_k.weight.data.normal_(mean=0.0, std=(config.num_heads * config.dim_qkv) ** -0.5)
            self.w_v.weight.data.normal_(mean=0.0, std=(config.num_heads * config.dim_qkv) ** -0.5)
        else:
            self.w_k.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)
            self.w_v.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)

        if self.has_relative_attention_bias:
            self.relative_attention_bias.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)

    # TODO implement relative position bias

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

        attention_values: SequenceInputEmbeds = self.reshape_to_head_insensitive(attention_values)

        attention_output: SequenceInputEmbeds = self.w_o(attention_values)

        return AttentionOutput(attention_output, (key, value))


class BaselineFeedforward(TransformerComponentBase):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)

        self.up_projection = Linear(config.dim_model, config.dim_feedforward, bias=False)
        self.down_projection = Linear(config.dim_feedforward, config.dim_model, bias=False)

    def init_weights(self) -> None:
        config = self.config

        self.up_projection.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)
        self.down_projection.weight.data.normal_(mean=0.0, std=config.dim_feedforward ** -0.5)

    def forward(self, input_embeds: SequenceInputEmbeds) -> SequenceInputEmbeds:
        config = self.config

        input_embeds: SequenceInputEmbeds = self.up_projection(input_embeds)
        input_embeds: SequenceInputEmbeds = relu(input_embeds)
        input_embeds: SequenceInputEmbeds = dropout(input_embeds, config.dropout_prob, training=self.training)
        input_embeds: SequenceInputEmbeds = self.down_projection(input_embeds)

        return input_embeds


class BaselineEncoder(EncoderBase):
    ATTENTION_CLS = BaselineAttention
    FEEDFORWARD_CLS = BaselineFeedforward


class BaselineDecoder(DecoderBase):
    ATTENTION_CLS = BaselineAttention
    FEEDFORWARD_CLS = BaselineFeedforward


class BaselineTransformer(TransformerBase):
    ENCODER_CLS = BaselineEncoder
    DECODER_CLS = BaselineDecoder
