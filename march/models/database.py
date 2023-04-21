from torch.nn import ParameterList

from march.models.utils import *
from march.models.baseline import *


DatabaseState = TensorType["L", "D_kv"]
DatabaseKeyValueStates = Tuple[DatabaseState, DatabaseState]


@dataclass
class DatabaseTransformerConfig(TransformerConfig):
    num_database_states: int = 512


class DatabaseEncoderBase(EncoderBase):
    def __init__(self, config: TransformerConfig) -> None:
        TransformerComponentBase.__init__(self, config)

        self.layernorms = ModuleList([LayerNorm(config) for _ in range(((config.num_layers // 2) * 3) + 1)])
        self.self_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(config, is_cross_attention=False) for _ in range(config.num_layers // 2)]
        )
        self.database_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(config, is_cross_attention=True) for _ in range(config.num_layers // 2)]
        )
        self.feedforward_layers: List[TransformerComponentBase] = ModuleList(
            [self.FEEDFORWARD_CLS(config) for _ in range(config.num_layers // 2)]
        )

    def forward(
        self,
        input_embeds: SequenceInputEmbeds,
        attention_mask: SequenceInputIds,
        database_key_value_states: DatabaseKeyValueStates,
    ) -> AttentionOutput:
        config: TransformerConfig = self.config

        encoder_key_value_states: List[KeyValueStates] = []
        for i in range(config.num_layers // 2):
            self_attention_layernorm: LayerNorm = self.layernorms[2 * i]
            self_attention_layer: AttentionBase = self.self_attention_layers[i]
            database_attention_layernorm: LayerNorm = self.layernorms[2 * i + 1]
            database_attention_layer: AttentionBase = self.database_attention_layers[i]
            feedforward_layernorm: LayerNorm = self.layernorms[2 * i + 2]
            feedforward_layer: TransformerComponentBase = self.feedforward_layers[i]

            normed_input_embeds: SequenceInputEmbeds = self_attention_layernorm(input_embeds)
            self_attention_output: AttentionOutput = self_attention_layer(normed_input_embeds, attention_mask)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(self_attention_output.input_embeds, p=config.dropout_prob, training=self.training)
            encoder_key_value_states.append(self_attention_output.key_value_states)

            normed_input_embeds: SequenceInputEmbeds = database_attention_layernorm(input_embeds)
            database_attention_output: AttentionOutput = database_attention_layer(normed_input_embeds, attention_mask=None, encoder_key_value_states=database_key_value_states)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(database_attention_output.input_embeds, p=config.dropout_prob, training=self.training)

            normed_input_embeds: SequenceInputEmbeds = feedforward_layernorm(input_embeds)
            feedforward_output: SequenceInputEmbeds = feedforward_layer(normed_input_embeds)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(feedforward_output, p=config.dropout_prob, training=self.training)

        input_embeds: SequenceInputEmbeds = self.layernorms[-1](input_embeds)

        return AttentionOutput(input_embeds=input_embeds, key_value_states=encoder_key_value_states)


class DatabaseDecoderBase(DecoderBase):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)

        self.layernorms = ModuleList([LayerNorm(config) for _ in range((config.num_layers * 2) + 1)])
        self.self_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(config, is_cross_attention=False) for _ in range(config.num_layers // 2)]
        )
        self.cross_attention_layers: List[AttentionBase] = ModuleList(
            [self.ATTENTION_CLS(config, is_cross_attention=True) for _ in range(config.num_layers // 2)]
        )
        self.database_attention_layers: List[AttentionBase] = ModuleList(
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
        database_key_value_states: DatabaseKeyValueStates,
    ) -> AttentionOutput:
        config: TransformerConfig = self.config

        for i in range(config.num_layers // 2):
            self_attention_layernorm: LayerNorm = self.layernorms[2 * i]
            self_attention_layer: AttentionBase = self.self_attention_layers[i]
            cross_attention_layernorm: LayerNorm = self.layernorms[2 * i + 1]
            cross_attention_key_value_states = encoder_key_value_states[i]
            cross_attention_layer: AttentionBase = self.cross_attention_layers[i]
            database_attention_layernorm: LayerNorm = self.layernorms[2 * i + 2]
            database_attention_layer: AttentionBase = self.database_attention_layers[i]
            feedforward_layernorm: LayerNorm = self.layernorms[2 * i + 3]
            feedforward_layer: TransformerComponentBase = self.feedforward_layers[i]

            normed_input_embeds: SequenceInputEmbeds = self_attention_layernorm(input_embeds)
            self_attention_output: AttentionOutput = self_attention_layer(normed_input_embeds, attention_mask)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(self_attention_output.input_embeds, p=config.dropout_prob, training=self.training)

            normed_input_embeds: SequenceInputEmbeds = cross_attention_layernorm(input_embeds)
            cross_attention_output: AttentionOutput = cross_attention_layer(normed_input_embeds, encoder_attention_mask, cross_attention_key_value_states)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(cross_attention_output.input_embeds, p=config.dropout_prob, training=self.training)

            normed_input_embeds: SequenceInputEmbeds = database_attention_layernorm(input_embeds)
            cross_attention_output: AttentionOutput = database_attention_layer(normed_input_embeds, attention_mask=None, encoder_key_value_states=database_key_value_states)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(cross_attention_output.input_embeds, p=config.dropout_prob, training=self.training)

            normed_input_embeds: SequenceInputEmbeds = feedforward_layernorm(input_embeds)
            feedforward_output: SequenceInputEmbeds = feedforward_layer(normed_input_embeds)
            input_embeds: SequenceInputEmbeds = input_embeds + dropout(feedforward_output, p=config.dropout_prob, training=self.training)

        input_embeds: SequenceInputEmbeds = self.layernorms[-1](input_embeds)

        return AttentionOutput(input_embeds=input_embeds, key_value_states=None)


class DatabaseEncoder(DatabaseEncoderBase):
    ATTENTION_CLS = BaselineAttention
    FEEDFORWARD_CLS = BaselineFeedforward


class DatabaseDecoder(DatabaseDecoderBase):
    ATTENTION_CLS = BaselineAttention
    FEEDFORWARD_CLS = BaselineFeedforward


class DatabaseTransformerBase(TransformerBase):
    def __init__(self, config: DatabaseTransformerConfig) -> None:
        TransformerComponentBase.__init__(self, config)

        self.config = config

        self.embedding: TensorType["D", "V"] = Linear(config.dim_model, VOCAB_SIZE, bias=False)
        self.position_encoding = self.POSITION_ENCODING_CLS(config)

        self.database_key_value_states: DatabaseKeyValueStates = ParameterList((
            Parameter(BFloat16Tensor(1, 1, config.num_database_states, config.dim_qkv)),
            Parameter(BFloat16Tensor(1, 1, config.num_database_states, config.dim_qkv)),
        ))

        self.encoder = self.ENCODER_CLS(config)
        self.decoder = self.DECODER_CLS(config)

        self.init_weights()

    def init_weights(self) -> None:
        self.embedding.weight.data.normal_(mean=0.0, std=1.0)

        self.database_key_value_states[0].data.normal_(mean=0.0, std=1.0)
        self.database_key_value_states[1].data.normal_(mean=0.0, std=1.0)

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
        encoder_outputs: AttentionOutput = self.encoder(input_embeds, attention_mask, self.database_key_value_states)

        decoder_input_embeds: SequenceInputEmbeds = embedding(decoder_input_ids, self.embedding.weight)
        decoder_input_embeds: SequenceInputEmbeds = self.position_encoding(decoder_input_embeds)
        decoder_attention_mask = self.create_decoder_attention_mask(decoder_input_ids)
        decoder_outputs: AttentionOutput = self.decoder(
            decoder_input_embeds, decoder_attention_mask, encoder_outputs.key_value_states, attention_mask, self.database_key_value_states
        )

        sequence_output = decoder_outputs.input_embeds
        sequence_output = sequence_output * (config.dim_model ** -0.5)

        lm_logits = self.embedding(sequence_output)

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(loss=loss, logits=lm_logits)


class DatabaseTransformer(DatabaseTransformerBase):
    POSITION_ENCODING_CLS = AbsolutePositionEncoding
    ENCODER_CLS = DatabaseEncoder
    DECODER_CLS = DatabaseDecoder
