from unittest import TestCase

from torch import bfloat16, long, randint
from torch.nn.functional import embedding

from march.experiments.tests import FCABSExperiment


class TestFCABS(TestCase):
    def setUp(self) -> None:
        self.exp = FCABSExperiment()
        self.model = self.exp.get_model().to(bfloat16)
        config = self.model.config

        # Create dummy inputs
        N, L = 1, 128
        self.input_ids = randint(0, config.vocab_size, (N, L), dtype=long)
        self.input_embeds = embedding(self.input_ids, self.model.embedding.weight).to(bfloat16)
        self.attention_mask = randint(0, 2, (N, L), dtype=bool)
        self.decoder_attention_mask = randint(0, 2, (N, L, L), dtype=bool)

    def test_encoder_basic(self) -> None:
        model = self.model
        config = model.config
        input_embeds = self.input_embeds
        attention_mask = self.attention_mask

        outputs = model.encoder(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
        )

        N, original_L, D = input_embeds.size()
        target_L = original_L - (config.L_drop * config.num_layers // 2)
        assert target_L > 0, target_L
        self.assertEqual(outputs.input_embeds.size(), (N, target_L, D))

    def test_training_has_no_custom_logs(self) -> None:
        self.model.train()

        _ = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            decoder_input_ids=self.input_ids,
            decoder_attention_mask=self.decoder_attention_mask,
            labels=self.input_ids,
        )

        custom_logs = self.model.get_custom_logs()

        self.assertNotIn("global_dropped_ids_by_layer", custom_logs)

    def test_logging(self) -> None:
        self.model.eval()

        _ = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            decoder_input_ids=self.input_ids,
            decoder_attention_mask=self.decoder_attention_mask,
            labels=self.input_ids,
        )

        custom_logs = self.model.get_custom_logs()

        self.assertIn("eval_global_dropped_ids_by_layer", custom_logs)

        value = custom_logs["eval_global_dropped_ids_by_layer"]
        self.assertIsInstance(value, list)
