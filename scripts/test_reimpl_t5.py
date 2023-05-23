import march  # redirect cache

from sys import argv

from os.path import exists

from unittest import TestCase, main as unittest_main, skipIf

from tqdm.auto import trange

from numpy import allclose, array, isclose, ndarray

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from torch import bfloat16, equal, finfo, long, manual_seed as set_torch_seed, rand, randint
from torch.cuda import device_count
from torch.optim import AdamW

from datasets import load_dataset

from march.experiments.baseline import BaselineSmallFullTrainExperiment, BaselineT5SmallExperiment


class TestExperimentMixin:
    NUM_STEPS = 100


class TestBaselineExperiment(TestExperimentMixin, BaselineSmallFullTrainExperiment):
    pass


class TestBaselineT5Experiment(TestExperimentMixin, BaselineT5SmallExperiment):
    pass


def read_train_loss(path: str) -> ndarray:
    event_accumulator = EventAccumulator(path)
    event_accumulator.Reload()

    scalars = event_accumulator.Scalars("train/loss")

    return array([s.value for s in scalars])


class TestReimplMatchT5Units(TestCase):
    SEED = 42

    def setUp(self) -> None:
        # Initialize reimpl and t5 models
        reimpl_exp = TestBaselineExperiment()
        t5_exp = TestBaselineT5Experiment()
        reimpl_model = reimpl_exp.get_model()
        t5_model = t5_exp.get_model()
        reimpl_model = reimpl_model.to(bfloat16)
        t5_model = t5_model.to(bfloat16)

        # Reset parameters
        reimpl_exp._call_init_weights(reimpl_model, self.SEED)
        t5_exp._call_init_weights(t5_model, self.SEED)

        self.reimpl_exp = reimpl_exp
        self.reimpl_model = reimpl_model
        self.t5_exp = t5_exp
        self.t5_model = t5_model

        # Create dummy inputs
        N, L = 2, 8
        D = reimpl_model.config.dim_model
        self.input_ids = randint(0, reimpl_model.config.vocab_size, (N, L), dtype=long)
        self.input_embeds = t5_model.shared(self.input_ids).to(bfloat16)
        self.encoder_hidden_state = rand((N, L, D), dtype=bfloat16)

        self.attention_mask = randint(0, 2, (N, L), dtype=bool)
        self.t5_attention_mask = t5_model.get_extended_attention_mask(~self.attention_mask, (N, L))
        self.decoder_attention_mask = randint(0, 2, (N, L, L), dtype=bool)
        self.t5_decoder_attention_mask = t5_model.get_extended_attention_mask(~self.decoder_attention_mask, (N, L, L))

    def test_weight_matching_basic(self) -> None:
        num_encoder_layers = self.reimpl_model.config.num_layers // 2
        # Sanity check weight matching using encoder selfattn q weight
        for i in range(num_encoder_layers):
            reimpl_weight = self.reimpl_model.encoder.self_attention_layers[i].w_q
            t5_weight = self.t5_model.encoder.block[i].layer[0].SelfAttention.q
            reimpl_weight = reimpl_weight.weight.data
            t5_weight = t5_weight.weight.data

            self.assertTrue(equal(reimpl_weight, t5_weight))

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
            attention_mask=~attention_mask,  # We don't use self.t5_attention_mask because encoder will do it for us
        )[0]

        self.assertTrue(equal(reimpl_encoder_outputs, t5_encoder_outputs))

        reimpl_encoder_outputs.mean().backward(retain_graph=True)
        t5_encoder_outputs.mean().backward(retain_graph=True)
        reimpl_selfattn = reimpl_model.encoder.self_attention_layers[0]
        t5_selfattn = t5_model.encoder.block[0].layer[0].SelfAttention
        self.assertTrue(equal(reimpl_selfattn.w_q.weight.grad, t5_selfattn.q.weight.grad))

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
            attention_mask=~decoder_attention_mask,  # Again, we don't use self.t5_decoder_attention_mask because decoder will do it for us
            encoder_hidden_states=encoder_hidden_state,
            encoder_attention_mask=~attention_mask,
        )[0]

        self.assertTrue(equal(reimpl_decoder_outputs, t5_decoder_outputs))

        reimpl_decoder_outputs.mean().backward(retain_graph=True)
        t5_decoder_outputs.mean().backward(retain_graph=True)
        reimpl_crossattn = reimpl_model.decoder.cross_attention_layers[0]
        t5_crossattn = t5_model.decoder.block[0].layer[1].EncDecAttention
        assert reimpl_crossattn.w_q.weight.grad is not None
        self.assertTrue(equal(reimpl_crossattn.w_q.weight.grad, t5_crossattn.q.weight.grad))

    def test_transformer(self) -> None:
        reimpl_model = self.reimpl_model
        t5_model = self.t5_model
        input_ids = self.input_ids
        attention_mask = self.attention_mask
        decoder_attention_mask = self.decoder_attention_mask

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

        # Check gradients
        reimpl_loss.backward()
        t5_loss.backward()

        reimpl_weight = reimpl_model.encoder.self_attention_layers[0].w_q
        t5_weight = t5_model.encoder.block[0].layer[0].SelfAttention.q
        reimpl_grad = reimpl_weight.weight.grad
        t5_grad = t5_weight.weight.grad
        self.assertTrue(equal(reimpl_grad, t5_grad))


@skipIf(device_count() == 0, "Need GPUs to run end to end experiment")
class TestReimplMatchT5(TestCase):
    SEED = 42  # Only used for this test case

    def test_integration(self) -> None:
        device = 7  # TODO Temporary, not sure how best to pass in a param with unittest lol
        move_formats = lambda t: t.to(f"cuda:{device}", bfloat16)

        reimpl_exp = TestBaselineExperiment()
        reimpl_model = reimpl_exp.get_model()
        reimpl_exp._call_init_weights(reimpl_model, self.SEED)
        move_formats(reimpl_model)
        t5_exp = TestBaselineT5Experiment()
        t5_model = t5_exp.get_model()
        t5_exp._call_init_weights(t5_model, self.SEED)
        move_formats(t5_model)

        create_optimizer = lambda model: AdamW(params=model.parameters(), lr=1e-4, weight_decay=1e-1)
        reimpl_optimizer = create_optimizer(reimpl_model)
        t5_optimizer = create_optimizer(t5_model)

        # We use .to_list to convert into a format readable by data collators
        tiny_dataset = load_dataset("hlillemark/c4_t5_100")["train"]
        batch_size = 2
        num_iters = 10

        inputs_to_cuda = lambda d: {k: v.cuda(device) for k, v in d.items()}
        reimpl_data_collator = reimpl_exp.get_data_collator(reimpl_exp.load_default_tokenizer())
        t5_data_collator = t5_exp.get_data_collator(t5_exp.load_default_tokenizer())

        for step in trange(num_iters, desc="Running through inputs"):
            data_start_idx, data_stop_idx = step * batch_size, (step + 1) * batch_size
            batch = tiny_dataset.select(range(data_start_idx, data_stop_idx)).to_list()

            reimpl_inputs = inputs_to_cuda(reimpl_data_collator(batch))
            t5_inputs = inputs_to_cuda(t5_data_collator(batch))

            set_torch_seed(self.SEED)
            reimpl_outputs = reimpl_model(**reimpl_inputs)
            set_torch_seed(self.SEED)
            t5_outputs = t5_model(**t5_inputs)

            self.assertTrue(equal(reimpl_outputs.logits, t5_outputs.logits), f"Failed on the {step+1} iteration")

            reimpl_loss = reimpl_outputs.loss
            t5_loss = t5_outputs.loss
            self.assertTrue(equal(reimpl_loss, t5_loss), f"Failed on the {step+1} iteration")

            reimpl_optimizer.zero_grad()
            t5_optimizer.zero_grad()

            # Check gradients
            reimpl_loss.backward()
            t5_loss.backward()

            reimpl_weight = reimpl_model.encoder.self_attention_layers[0].w_q
            t5_weight = t5_model.encoder.block[0].layer[0].SelfAttention.q
            reimpl_grad = reimpl_weight.weight.grad
            t5_grad = t5_weight.weight.grad
            self.assertTrue(equal(reimpl_grad, t5_grad), f"Failed on the {step+1} iteration")

            reimpl_optimizer.step()
            t5_optimizer.step()

    def test_end_to_end_train(self) -> None:
        reimpl_exp = TestBaselineExperiment()
        if not exists(reimpl_exp.output_dir): reimpl_exp.train()

        t5_exp = TestBaselineT5Experiment()
        if not exists(t5_exp.output_dir): t5_exp.train()

        reimpl_train_loss = read_train_loss(reimpl_exp.output_dir)
        t5_train_loss = read_train_loss(t5_exp.output_dir)

        atol = finfo(bfloat16).eps
        self.assertTrue(allclose(reimpl_train_loss, t5_train_loss, atol=atol))


def print_e2e_train_losses():
    reimpl_exp = TestBaselineExperiment()
    t5_exp = TestBaselineT5Experiment()

    reimpl_train_loss = read_train_loss(reimpl_exp.output_dir)
    t5_train_loss = read_train_loss(t5_exp.output_dir)
    diff = reimpl_train_loss - t5_train_loss

    atol = finfo(bfloat16).eps
    values_isclose = isclose(reimpl_train_loss, t5_train_loss, atol=atol)

    print(reimpl_train_loss, t5_train_loss, diff, values_isclose, sep="\n")


if __name__ == "__main__":
    unittest_args = argv[:1]
    unittest_main(argv=unittest_args)
