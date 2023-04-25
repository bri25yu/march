from torch.backends.cuda import sdp_kernel

# https://pytorch.org/docs/master/backends.html#torch.backends.cuda.enable_flash_sdp
# Enable flash attention only
sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)

from torch.nn import MultiheadAttention

from march.models.utils import *
from march.models.baseline import *


class FastAttention(TransformerComponentBase):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)

        self.attention = MultiheadAttention(
            embed_dim=config.dim_model,
            num_heads=config.num_heads,
            dropout=config.dropout_prob,
            bias=False,
            batch_first=True,
        )

    def init_weights(self) -> None:
        config = self.config

        self.attention.in_proj_weight.weight.data.normal_(mean=0.0, std=config.dim_model ** -0.5)
        self.attention.out_proj.weight.data.normal(mean=0.0, std=(config.num_heads * config.dim_qkv) ** -0.5)

    def forward(
        self,
        input_embeds: SequenceInputEmbeds,
        attention_mask: SequenceInputIds=None,
        encoder_input_embeds: SequenceInputEmbeds=None,
    ) -> AttentionOutput:
        if encoder_input_embeds is not None:  # Cross attention
            key = value = encoder_input_embeds
        else:  # Self attention
            key = value = input_embeds

        attention_output = self.attention(
            query=input_embeds, key=key, value=value,
            attn_mask=attention_mask, is_causal=attention_mask is None,
            need_weights=False,
        )

        return AttentionOutput(attention_output, input_embeds)


class FastFeedforward(BaselineFeedforward):
    def __init__(self, config: TransformerConfig) -> None:
        TransformerComponentBase.__init__(self, config)

        hidden_dim = round(config.dim_feedforward / 64) * 64
        self.up_projection = Linear(config.dim_model, hidden_dim, bias=False)
        self.down_projection = Linear(hidden_dim, config.dim_model, bias=False)


class FastEncoderBase(EncoderBase):
    def __init__(self, config: TransformerConfig) -> None:
        TransformerComponentBase.__init__(self, config)

        self.layernorms = ModuleList([LayerNorm(config) for _ in range(config.num_layers + 1)])
        self.self_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(config) for _ in range(config.num_layers // 2)]
        )
        self.feedforward_layers: List[TransformerComponentBase] = ModuleList(
            [self.FEEDFORWARD_CLS(config) for _ in range(config.num_layers // 2)]
        )

    def forward(self, input_embeds: SequenceInputEmbeds, attention_mask: SequenceInputIds) -> AttentionOutput:
        config: TransformerConfig = self.config

        encoder_input_embeds: List[SequenceInputEmbeds] = []
        for i in range(config.num_layers // 2):
            self_attention_layernorm: LayerNorm = self.layernorms[2 * i]
            self_attention_layer: AttentionBase = self.self_attention_layers[i]
            feedforward_layernorm: LayerNorm = self.layernorms[2 * i + 1]
            feedforward_layer: TransformerComponentBase = self.feedforward_layers[i]

            normed_input_embeds: SequenceInputEmbeds = self_attention_layernorm(input_embeds)
            self_attention_output: AttentionOutput = self_attention_layer(normed_input_embeds, attention_mask)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(self_attention_output.input_embeds, p=config.dropout_prob, training=self.training)
            encoder_input_embeds.append(self_attention_output.key_value_states)

            normed_input_embeds: SequenceInputEmbeds = feedforward_layernorm(input_embeds)
            feedforward_output: SequenceInputEmbeds = feedforward_layer(normed_input_embeds)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(feedforward_output, p=config.dropout_prob, training=self.training)

        input_embeds: SequenceInputEmbeds = self.layernorms[-1](input_embeds)

        return AttentionOutput(input_embeds=input_embeds, key_value_states=encoder_input_embeds)


class FastDecoderBase(DecoderBase):
    def __init__(self, config: TransformerConfig) -> None:
        TransformerComponentBase.__init__(self, config)

        self.layernorms = ModuleList([LayerNorm(config) for _ in range(((config.num_layers // 2) * 3) + 1)])
        self.self_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(config) for _ in range(config.num_layers // 2)]
        )
        self.cross_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(config) for _ in range(config.num_layers // 2)]
        )
        self.feedforward_layers: List[TransformerComponentBase] = ModuleList(
            [self.FEEDFORWARD_CLS(config) for _ in range(config.num_layers // 2)]
        )

    def forward(
        self,
        input_embeds: SequenceInputEmbeds,
        encoder_input_embeds: List[SequenceInputEmbeds],
        encoder_attention_mask: SequenceInputIds,
    ) -> AttentionOutput:
        config: TransformerConfig = self.config

        for i in range(config.num_layers // 2):
            self_attention_layernorm: LayerNorm = self.layernorms[2 * i]
            self_attention_layer: AttentionBase = self.self_attention_layers[i]
            cross_attention_layernorm: LayerNorm = self.layernorms[2 * i + 1]
            cross_attention_encoder_input_embeds = encoder_input_embeds[i]
            cross_attention_layer: AttentionBase = self.cross_attention_layers[i]
            feedforward_layernorm: LayerNorm = self.layernorms[2 * i + 2]
            feedforward_layer: TransformerComponentBase = self.feedforward_layers[i]

            normed_input_embeds: SequenceInputEmbeds = self_attention_layernorm(input_embeds)
            self_attention_output: AttentionOutput = self_attention_layer(normed_input_embeds)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(self_attention_output.input_embeds, p=config.dropout_prob, training=self.training)

            normed_input_embeds: SequenceInputEmbeds = cross_attention_layernorm(input_embeds)
            cross_attention_output: AttentionOutput = cross_attention_layer(normed_input_embeds, encoder_attention_mask, cross_attention_encoder_input_embeds)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(cross_attention_output.input_embeds, p=config.dropout_prob, training=self.training)

            normed_input_embeds: SequenceInputEmbeds = feedforward_layernorm(input_embeds)
            feedforward_output: SequenceInputEmbeds = feedforward_layer(normed_input_embeds)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(feedforward_output, p=config.dropout_prob, training=self.training)

        input_embeds: SequenceInputEmbeds = self.layernorms[-1](input_embeds)

        return AttentionOutput(input_embeds=input_embeds, key_value_states=None)


class FastTransformerBase(TransformerBase):
    def __init__(self, config: TransformerConfig) -> None:
        TransformerComponentBase.__init__(self, config)

        self.config = config

        # https://twitter.com/cHHillee/status/1630274804795445248
        embedding_size = round(VOCAB_SIZE / 64) * 64
        self.embedding: TensorType["D", "V"] = Linear(config.dim_model, embedding_size, bias=False)
        self.position_encoding = self.POSITION_ENCODING_CLS(config)

        self.encoder = self.ENCODER_CLS(config)
        self.decoder = self.DECODER_CLS(config)

    def forward(
        self,
        input_ids: SequenceInputIds,
        attention_mask: SequenceInputIds,
        decoder_input_ids: SequenceInputIds,
        labels: SequenceInputIds,
    ) -> Seq2SeqLMOutput:
        config = self.config

        input_embeds: SequenceInputEmbeds = embedding(input_ids, self.embedding.weight)
        input_embeds: SequenceInputEmbeds = self.position_encoding(input_embeds)
        encoder_outputs: AttentionOutput = self.encoder(input_embeds, attention_mask)

        decoder_input_embeds: SequenceInputEmbeds = embedding(decoder_input_ids, self.embedding.weight)
        decoder_input_embeds: SequenceInputEmbeds = self.position_encoding(decoder_input_embeds)
        decoder_outputs: AttentionOutput = self.decoder(
            decoder_input_embeds, encoder_outputs.key_value_states, attention_mask,
        )

        sequence_output = decoder_outputs.input_embeds
        sequence_output = sequence_output * (config.dim_model ** -0.5)

        lm_logits = self.embedding(sequence_output)

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(loss=loss, logits=lm_logits)


class FastEncoder(FastEncoderBase):
    ATTENTION_CLS = FastAttention
    FEEDFORWARD_CLS = FastFeedforward


class FastDecoder(FastDecoderBase):
    ATTENTION_CLS = FastAttention
    FEEDFORWARD_CLS = FastFeedforward


class FastTransformer(FastTransformerBase):
    POSITION_ENCODING_CLS = AbsolutePositionEncoding
    ENCODER_CLS = FastEncoder
    DECODER_CLS = FastDecoder
