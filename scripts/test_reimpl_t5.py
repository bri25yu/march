from sys import argv

from unittest import TestCase, main as unittest_main, skipIf

from numpy import array, ndarray

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from torch import equal, long, manual_seed as set_torch_seed, rand, randint, ones
from torch.cuda import device_count

from march.experiments.baseline import BaselineExperiment, BaselineT5Experiment


class TestExperimentMixin:
    NUM_STEPS = 5


class TestBaselineExperiment(TestExperimentMixin, BaselineExperiment):
    pass


class TestBaselineT5Experiment(TestExperimentMixin, BaselineT5Experiment):
    pass


def read_train_loss(path: str) -> ndarray:
    event_accumulator = EventAccumulator(path)
    event_accumulator.Reload()

    scalars = event_accumulator.Scalars("train/loss")

    return array([s.value for s in scalars])


@skipIf(device_count() != 0, "Skipping unit tests with GPUs")
class TestReimplMatchT5Units(TestCase):
    SEED = 42

    def setUp(self) -> None:
        # Initialize reimpl and t5 models
        reimpl_exp = TestBaselineExperiment()
        t5_exp = TestBaselineT5Experiment()
        reimpl_model = reimpl_exp.get_model()
        t5_model = t5_exp.get_model()

        # Reset parameters
        reimpl_exp._call_init_weights(reimpl_model, self.SEED)
        t5_exp._call_init_weights(t5_model, self.SEED)

        self.reimpl_exp = reimpl_exp
        self.reimpl_model = reimpl_model
        self.t5_exp = t5_exp
        self.t5_model = t5_model

        # Create dummy inputs
        N, L = 2, 128
        D = reimpl_model.config.dim_model
        self.input_ids = randint(0, reimpl_model.config.vocab_size, (N, L), dtype=long)
        self.input_embeds = t5_model.shared(self.input_ids)
        self.encoder_hidden_state = rand((N, L, D))

        self.attention_mask = ones((N, L), dtype=long)
        self.t5_attention_mask = t5_model.get_extended_attention_mask(1.0 - self.attention_mask, (N, L))
        self.decoder_attention_mask = reimpl_model.create_decoder_attention_mask(self.input_ids)
        self.t5_decoder_attention_mask = t5_model.get_extended_attention_mask(1.0 - self.decoder_attention_mask, (N, L))

    def test_weight_matching_basic(self) -> None:
        # Sanity check weight matching using encoder selfattn q weight
        for i in range(12):
            reimpl_weight = self.reimpl_model.encoder.self_attention_layers[i].w_q
            t5_weight = self.t5_model.encoder.block[i].layer[0].SelfAttention.q
            reimpl_weight = reimpl_weight.weight.data
            t5_weight = t5_weight.weight.data

            self.assertTrue(equal(reimpl_weight, t5_weight))

    def test_selfattn(self) -> None:
        reimpl_model = self.reimpl_model
        t5_model = self.t5_model
        input_embeds = self.input_embeds
        attention_mask = self.attention_mask
        t5_attention_mask = self.t5_attention_mask

        for i in range(12):
            reimpl_selfattn = reimpl_model.encoder.self_attention_layers[i]
            t5_selfattn = t5_model.encoder.block[i].layer[0].SelfAttention

            set_torch_seed(self.SEED)
            reimpl_outputs = reimpl_selfattn(input_embeds, attention_mask).input_embeds
            set_torch_seed(self.SEED)
            t5_outputs = t5_selfattn(input_embeds, t5_attention_mask)[0]

            self.assertTrue(equal(reimpl_outputs, t5_outputs))

    def test_crossattn(self) -> None:
        reimpl_model = self.reimpl_model
        t5_model = self.t5_model
        input_embeds = self.input_embeds
        decoder_attention_mask = self.decoder_attention_mask
        t5_decoder_attention_mask = self.t5_decoder_attention_mask
        encoder_hidden_state = self.encoder_hidden_state

        for i in range(12):
            reimpl_crossattn = reimpl_model.decoder.cross_attention_layers[i]
            t5_crossattn = t5_model.decoder.block[i].layer[1].EncDecAttention

            set_torch_seed(self.SEED)
            reimpl_outputs = reimpl_crossattn(input_embeds, decoder_attention_mask, encoder_hidden_state=encoder_hidden_state).input_embeds
            set_torch_seed(self.SEED)
            t5_outputs = t5_crossattn(input_embeds, t5_decoder_attention_mask, key_value_states=encoder_hidden_state)[0]

            self.assertTrue(equal(reimpl_outputs, t5_outputs))

    def test_encoder(self) -> None:
        reimpl_model = self.reimpl_model
        t5_model = self.t5_model
        input_embeds = self.input_embeds
        attention_mask = self.attention_mask

        set_torch_seed(self.SEED)
        reimpl_encoder_outputs = reimpl_model.encoder(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
        ).input_embeds

        set_torch_seed(self.SEED)
        t5_encoder_outputs = t5_model.encoder(
            inputs_embeds=input_embeds,
            attention_mask=1.0 - attention_mask,  # We don't use self.t5_attention_mask because encoder will do it for us
        )[0]

        self.assertTrue(equal(reimpl_encoder_outputs, t5_encoder_outputs))

    def test_decoder(self) -> None:
        reimpl_model = self.reimpl_model
        t5_model = self.t5_model
        input_embeds = self.input_embeds
        attention_mask = self.attention_mask
        decoder_attention_mask = self.decoder_attention_mask
        encoder_hidden_state = self.encoder_hidden_state

        set_torch_seed(self.SEED)
        reimpl_decoder_outputs = reimpl_model.decoder(
            input_embeds=input_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_state=encoder_hidden_state,
            encoder_attention_mask=attention_mask,
        ).input_embeds
        set_torch_seed(self.SEED)
        t5_decoder_outputs = t5_model.decoder(
            inputs_embeds=input_embeds,
            attention_mask=1.0 - decoder_attention_mask,  # Again, we don't use self.t5_decoder_attention_mask because decoder will do it for us
            encoder_hidden_states=encoder_hidden_state,
            encoder_attention_mask=1.0 - attention_mask,
        )[0]

        self.assertTrue(equal(reimpl_decoder_outputs, t5_decoder_outputs))

    def test_single_step(self) -> None:
        reimpl_model = self.reimpl_model
        t5_model = self.t5_model
        input_ids = self.input_ids
        attention_mask = self.attention_mask

        set_torch_seed(self.SEED)
        reimpl_outputs = reimpl_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=input_ids,
            labels=input_ids,
        )
        set_torch_seed(self.SEED)
        t5_outputs = t5_model(
            input_ids=input_ids,
            attention_mask=1.0 - attention_mask,
            decoder_input_ids=input_ids,
            labels=input_ids,
        )

        reimpl_logits = reimpl_outputs.logits
        t5_logits = t5_outputs.logits
        self.assertTrue(equal(reimpl_logits, t5_logits))

        reimpl_loss = reimpl_outputs.loss
        t5_loss = t5_outputs.loss
        self.assertTrue(equal(reimpl_loss, t5_loss))

        # Check gradients
        reimpl_loss.backward()
        t5_loss.backward()

        reimpl_weight = reimpl_model.encoder.self_attention_layers[0].w_q
        t5_weight = t5_model.encoder.block[0].layer[0].SelfAttention.q
        reimpl_grad = reimpl_weight.weight.grad
        t5_grad = t5_weight.weight.grad
        self.assertTrue(equal(reimpl_grad, t5_grad))


@skipIf(device_count() == 0, "Need GPUs to run end to end experiment")
class TestReimplMatchT5EndToEnd(TestCase):
    def test_end_to_end_train(self) -> None:
        reimpl_exp = TestBaselineExperiment()
        reimpl_exp.train()

        t5_exp = TestBaselineT5Experiment()
        t5_exp.train()

        reimpl_train_loss = read_train_loss(reimpl_exp.output_dir)
        t5_train_loss = read_train_loss(t5_exp.output_dir)

        self.assertTrue((reimpl_train_loss == t5_train_loss).all())


if __name__ == "__main__":
    unittest_args = argv[:1]
    unittest_main(argv=unittest_args)
