from unittest import TestCase

from torch import bfloat16
from torch.nn.functional import embedding

from datasets import load_dataset

from tests.fcabs.experiment_mixins import TestFCABSExperiment


class TestFCABSUnits(TestCase):
    def setUp(self) -> None:
        self.exp = TestFCABSExperiment()
        self.model = self.exp.get_model().to(bfloat16)

        N = 2
        data_collator = self.exp.get_data_collator(self.exp.load_default_tokenizer())
        tiny_dataset = load_dataset("hlillemark/c4_t5_100")["train"]
        inputs = tiny_dataset.select(range(N)).to_list()
        self.inputs = data_collator(inputs)

    def test_encoder_basic(self) -> None:
        model = self.model
        config = model.config
        inputs = self.inputs

        model.train()

        input_embeds = embedding(inputs["input_ids"], model.embedding.weight)
        encoder_inputs = {
            "input_embeds": input_embeds,
            "attention_mask": inputs["attention_mask"],
        }

        outputs = model.encoder(**encoder_inputs)

        N, original_L = inputs["input_ids"].size()
        D = config.dim_model
        target_L = original_L - (config.L_drop * config.num_layers // 2)
        assert target_L > 0, target_L
        self.assertEqual(outputs.input_embeds.size(), (N, target_L, D))

    def test_training_has_no_logs(self) -> None:
        model = self.model
        inputs = self.inputs

        model.train()

        model_outputs = model(**inputs)

        self.assertIsNone(model_outputs.dropped_ids)

    def test_eval_has_logs(self) -> None:
        model = self.model
        inputs = self.inputs
        config = model.config

        model.eval()

        model_outputs = model(**inputs)

        dropped_ids = model_outputs.dropped_ids
        self.assertIsNotNone(dropped_ids)
        expected_size = (inputs["input_ids"].size()[0], config.num_layers // 2, config.L_drop)  # (N, N_L, L_drop)
        self.assertEqual(dropped_ids.size(), expected_size)
