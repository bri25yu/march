from unittest import TestCase

from torch import equal, manual_seed as set_torch_seed

from tests.reimpl_t5.match_weights import *
from tests.reimpl_t5.component_test_mixins import *


class TestReimplMatchT5SmallComponents(ComponentTestMixin, TestCase):
    def test_ff(self) -> None:
        reimpl_model = self.reimpl_model
        t5_model = self.t5_model
        input_embeds = self.input_embeds
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
                reimpl_outputs = reimpl_ff(input_embeds)
                set_torch_seed(self.SEED)
                t5_outputs = t5_ff(input_embeds)

                self.assertTrue(equal(reimpl_outputs, t5_outputs))

                reimpl_outputs.mean().backward(retain_graph=True)
                t5_outputs.mean().backward(retain_graph=True)

                reimpl_weight = reimpl_ff.up_projection
                t5_weight = t5_ff.wi
                self.assertTrue(equal(reimpl_weight.weight.grad, t5_weight.weight.grad))

    def test_selfattn(self) -> None:
        reimpl_model = self.reimpl_model
        t5_model = self.t5_model
        input_embeds = self.input_embeds
        attention_mask = self.attention_mask
        t5_attention_mask = self.t5_attention_mask
        num_encoder_layers = self.reimpl_model.config.num_layers // 2

        for i in range(num_encoder_layers):
            reimpl_selfattn = reimpl_model.encoder.self_attention_layers[i]
            t5_selfattn = t5_model.encoder.block[i].layer[0].SelfAttention

            set_torch_seed(self.SEED)
            reimpl_outputs = reimpl_selfattn(input_embeds, attention_mask).input_embeds
            set_torch_seed(self.SEED)
            t5_outputs = t5_selfattn(input_embeds, t5_attention_mask)[0]

            self.assertTrue(equal(reimpl_outputs, t5_outputs))

            reimpl_outputs.mean().backward(retain_graph=True)
            t5_outputs.mean().backward(retain_graph=True)
            reimpl_weight = reimpl_selfattn.w_q
            t5_weight = t5_selfattn.q
            self.assertTrue(equal(reimpl_weight.weight.grad, t5_weight.weight.grad))

    def test_crossattn(self) -> None:
        reimpl_model = self.reimpl_model
        t5_model = self.t5_model
        input_embeds = self.input_embeds
        decoder_attention_mask = self.decoder_attention_mask
        t5_decoder_attention_mask = self.t5_decoder_attention_mask
        encoder_hidden_state = self.encoder_hidden_state
        num_encoder_layers = self.reimpl_model.config.num_layers // 2

        for i in range(num_encoder_layers):
            reimpl_crossattn = reimpl_model.decoder.cross_attention_layers[i]
            t5_crossattn = t5_model.decoder.block[i].layer[1].EncDecAttention

            set_torch_seed(self.SEED)
            reimpl_outputs = reimpl_crossattn(input_embeds, decoder_attention_mask, encoder_hidden_state=encoder_hidden_state).input_embeds
            set_torch_seed(self.SEED)
            t5_outputs = t5_crossattn(input_embeds, t5_decoder_attention_mask, key_value_states=encoder_hidden_state)[0]

            self.assertTrue(equal(reimpl_outputs, t5_outputs))

            reimpl_outputs.mean().backward(retain_graph=True)
            t5_outputs.mean().backward(retain_graph=True)
            reimpl_weight = reimpl_crossattn.w_q
            t5_weight = t5_crossattn.q
            self.assertTrue(equal(reimpl_weight.weight.grad, t5_weight.weight.grad))
