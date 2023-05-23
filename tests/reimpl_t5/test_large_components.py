from unittest import TestCase

from torch import equal, manual_seed as set_torch_seed
from torch.optim import AdamW

from tests.reimpl_t5.match_weights import *
from tests.reimpl_t5.component_test_mixins import *


class TestReimplMatchT5LargeComponents(ComponentTestMixin, TestCase):
    def test_encoder(self) -> None:
        reimpl_model = self.reimpl_model
        t5_model = self.t5_model
        reimpl_input_embeds = self.reimpl_input_embeds
        t5_input_embeds = self.t5_input_embeds
        attention_mask = self.attention_mask

        set_torch_seed(self.SEED)
        reimpl_encoder_outputs = reimpl_model.encoder(
            input_embeds=reimpl_input_embeds,
            attention_mask=attention_mask,
        ).input_embeds

        set_torch_seed(self.SEED)
        t5_encoder_outputs = t5_model.encoder(
            inputs_embeds=t5_input_embeds,
            attention_mask=~attention_mask,  # We don't use self.t5_attention_mask because encoder will do it for us
        )[0]

        self.assertTrue(equal(reimpl_encoder_outputs, t5_encoder_outputs))

        reimpl_encoder_outputs.mean().backward(retain_graph=True)
        t5_encoder_outputs.mean().backward(retain_graph=True)
        assert_grad_equal(reimpl_model, t5_model)
        reimpl_model.zero_grad()
        t5_model.zero_grad()

    def test_decoder(self) -> None:
        reimpl_model = self.reimpl_model
        t5_model = self.t5_model
        reimpl_input_embeds = self.reimpl_input_embeds
        t5_input_embeds = self.t5_input_embeds
        attention_mask = self.attention_mask
        decoder_attention_mask = self.decoder_attention_mask
        encoder_hidden_state = self.encoder_hidden_state

        set_torch_seed(self.SEED)
        reimpl_decoder_outputs = reimpl_model.decoder(
            input_embeds=reimpl_input_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_state=encoder_hidden_state,
            encoder_attention_mask=attention_mask,
        ).input_embeds

        set_torch_seed(self.SEED)
        t5_decoder_outputs = t5_model.decoder(
            inputs_embeds=t5_input_embeds,
            attention_mask=~decoder_attention_mask,  # Again, we don't use self.t5_decoder_attention_mask because decoder will do it for us
            encoder_hidden_states=encoder_hidden_state,
            encoder_attention_mask=~attention_mask,
        )[0]

        self.assertTrue(equal(reimpl_decoder_outputs, t5_decoder_outputs))

        reimpl_decoder_outputs.mean().backward(retain_graph=True)
        t5_decoder_outputs.mean().backward(retain_graph=True)
        assert_grad_equal(reimpl_model, t5_model)
        reimpl_model.zero_grad()
        t5_model.zero_grad()

    def test_transformer(self) -> None:
        reimpl_model = self.reimpl_model
        t5_model = self.t5_model
        input_ids = self.input_ids
        attention_mask = self.attention_mask
        decoder_attention_mask = self.decoder_attention_mask

        create_optimizer = lambda model: AdamW(params=model.parameters(), lr=1e-4, weight_decay=1e-1)
        reimpl_optimizer = create_optimizer(reimpl_model)
        t5_optimizer = create_optimizer(t5_model)

        set_torch_seed(self.SEED)
        reimpl_outputs = reimpl_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=input_ids,
        )
        set_torch_seed(self.SEED)
        t5_outputs = t5_model(
            input_ids=input_ids,
            attention_mask=~attention_mask,
            decoder_input_ids=input_ids,
            decoder_attention_mask=~decoder_attention_mask,
            labels=input_ids,
        )

        reimpl_logits = reimpl_outputs.logits
        t5_logits = t5_outputs.logits
        self.assertTrue(equal(reimpl_logits, t5_logits))

        reimpl_loss = reimpl_outputs.loss
        t5_loss = t5_outputs.loss
        self.assertTrue(equal(reimpl_loss, t5_loss))

        reimpl_optimizer.zero_grad()
        t5_optimizer.zero_grad()
        reimpl_loss.backward()
        t5_loss.backward()
        assert_grad_equal(reimpl_model, t5_model)

        reimpl_optimizer.step()
        t5_optimizer.step()
        assert_weight_equal(reimpl_model, t5_model)
