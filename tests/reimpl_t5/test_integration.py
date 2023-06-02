from unittest import TestCase, skipIf

from tqdm.auto import trange

from torch import bfloat16, equal, manual_seed as set_torch_seed
from torch.cuda import device_count
from torch.optim import AdamW

from datasets import load_dataset

from tests.reimpl_t5.match_weights import *
from tests.reimpl_t5.experiment_mixins import *


@skipIf(device_count() == 0, "Need GPUs to run integration experiment")
class TestReimplMatchT5Integration(TestCase):
    SEED = 42  # Only used for this test case

    def test_integration(self) -> None:
        device = (
            7  # TODO Temporary, not sure how best to pass in a param with unittest lol
        )
        move_formats = lambda t: t.to(f"cuda:{device}", bfloat16)

        reimpl_exp = TestBaselineExperiment()
        reimpl_model = reimpl_exp.get_model()
        reimpl_exp._call_init_weights(reimpl_model, self.SEED)
        move_formats(reimpl_model)
        t5_exp = TestBaselineT5Experiment()
        t5_model = t5_exp.get_model()
        t5_exp._call_init_weights(t5_model, self.SEED)
        move_formats(t5_model)

        create_optimizer = lambda model: AdamW(
            params=model.parameters(), lr=1e-4, weight_decay=1e-1
        )
        reimpl_optimizer = create_optimizer(reimpl_model)
        t5_optimizer = create_optimizer(t5_model)

        # We use .to_list to convert into a format readable by data collators
        tiny_dataset = load_dataset("hlillemark/c4_t5_100")["train"]
        batch_size = 2
        num_iters = 10

        inputs_to_cuda = lambda d: {k: v.cuda(device) for k, v in d.items()}
        reimpl_data_collator = reimpl_exp.get_data_collator(
            reimpl_exp.load_default_tokenizer()
        )
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

            self.assertTrue(
                equal(reimpl_outputs.logits, t5_outputs.logits),
                f"Failed on iteration {step+1}",
            )

            reimpl_loss = reimpl_outputs.loss
            t5_loss = t5_outputs.loss
            self.assertTrue(
                equal(reimpl_loss, t5_loss), f"Failed on iteration {step+1}"
            )

            reimpl_optimizer.zero_grad()
            t5_optimizer.zero_grad()

            reimpl_loss.backward()
            t5_loss.backward()
            assert_grad_equal(reimpl_model, t5_model)

            reimpl_optimizer.step()
            t5_optimizer.step()
            assert_weight_equal(reimpl_model, t5_model)
