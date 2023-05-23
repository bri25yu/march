from unittest import TestCase

from torch import equal, manual_seed as set_torch_seed

from tests.reimpl_t5.match_weights import *
from tests.reimpl_t5.component_test_mixins import *


class TestReimplMatchT5SmallComponents(ComponentTestMixin, TestCase):
    def test_ff(self) -> None:
        reimpl_model = self.reimpl_model
        t5_model = self.t5_model
        reimpl_input_embeds = self.reimpl_input_embeds
        t5_input_embeds = self.t5_input_embeds
        num_encoder_layers = self.reimpl_model.config.num_layers // 2

        bases = [
            (reimpl_model.encoder, t5_model.encoder),
            (reimpl_model.decoder, t5_model.decoder),
        ]

        for reimpl_base, t5_base in bases:
            for i in range(num_encoder_layers):
                reimpl_ff = reimpl_base.feedforward_layers[i]
                t5_ff = t5_base.block[i].layer[-1].DenseReluDense

                set_torch_seed(self.SEED)
                reimpl_outputs = reimpl_ff(reimpl_input_embeds)
                set_torch_seed(self.SEED)
                t5_outputs = t5_ff(t5_input_embeds)

                self.assertTrue(equal(reimpl_outputs, t5_outputs))

                reimpl_outputs.mean().backward(retain_graph=True)
                t5_outputs.mean().backward(retain_graph=True)

                assert_grad_equal(reimpl_model, t5_model)
                reimpl_model.zero_grad()
                t5_model.zero_grad()

    def test_selfattn(self) -> None:
        reimpl_model = self.reimpl_model
        t5_model = self.t5_model
        initial_reimpl_input_embeds = self.reimpl_input_embeds
        initial_t5_input_embeds = self.t5_input_embeds
        attention_mask = self.attention_mask
        t5_attention_mask = self.t5_attention_mask
        num_encoder_layers = self.reimpl_model.config.num_layers // 2

        current_reimpl_input_embeds = initial_reimpl_input_embeds
        current_t5_input_embeds = initial_t5_input_embeds
        current_reimpl_position_bias = None
        current_t5_position_bias = None
        for i in range(num_encoder_layers):
            reimpl_selfattn = reimpl_model.encoder.self_attention_layers[i]
            t5_selfattn = t5_model.encoder.block[i].layer[0].SelfAttention

            set_torch_seed(self.SEED)
            reimpl_outputs = reimpl_selfattn(current_reimpl_input_embeds, attention_mask, position_bias=current_reimpl_position_bias)
            set_torch_seed(self.SEED)
            t5_outputs = t5_selfattn(current_t5_input_embeds, t5_attention_mask, position_bias=current_t5_position_bias)

            reimpl_output_logits = reimpl_outputs.input_embeds
            t5_output_logits = t5_outputs[0]
            self.assertTrue(equal(reimpl_output_logits, t5_output_logits))

            current_reimpl_input_embeds = reimpl_output_logits
            current_t5_input_embeds = t5_output_logits
            current_reimpl_position_bias = reimpl_outputs.position_bias
            current_t5_position_bias = t5_outputs[2]

        current_reimpl_input_embeds.mean().backward()
        current_t5_input_embeds.mean().backward()
        assert_grad_equal(reimpl_model, t5_model)
        reimpl_model.zero_grad()
        t5_model.zero_grad()

    def test_crossattn(self) -> None:
        reimpl_model = self.reimpl_model
        t5_model = self.t5_model
        initial_reimpl_input_embeds = self.reimpl_input_embeds
        initial_t5_input_embeds = self.t5_input_embeds
        decoder_attention_mask = self.decoder_attention_mask
        t5_decoder_attention_mask = self.t5_decoder_attention_mask
        encoder_hidden_state = self.encoder_hidden_state
        num_encoder_layers = self.reimpl_model.config.num_layers // 2

        current_reimpl_input_embeds = initial_reimpl_input_embeds
        current_t5_input_embeds = initial_t5_input_embeds
        for i in range(num_encoder_layers):
            reimpl_crossattn = reimpl_model.decoder.cross_attention_layers[i]
            t5_crossattn = t5_model.decoder.block[i].layer[1].EncDecAttention

            set_torch_seed(self.SEED)
            reimpl_outputs = reimpl_crossattn(current_reimpl_input_embeds, decoder_attention_mask, encoder_hidden_state=encoder_hidden_state)
            set_torch_seed(self.SEED)
            t5_outputs = t5_crossattn(current_t5_input_embeds, t5_decoder_attention_mask, key_value_states=encoder_hidden_state)

            reimpl_output_logits = reimpl_outputs.input_embeds
            t5_output_logits = t5_outputs[0]
            self.assertTrue(equal(reimpl_output_logits, t5_output_logits))

            current_reimpl_input_embeds = reimpl_output_logits
            current_t5_input_embeds = t5_output_logits

        current_reimpl_input_embeds.mean().backward()
        current_t5_input_embeds.mean().backward()
        assert_grad_equal(reimpl_model, t5_model)
        reimpl_model.zero_grad()
        t5_model.zero_grad()
