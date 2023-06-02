import march  # Redirect cache. This is necessary because this file might be run as a standalone

from sys import argv

from os.path import exists

from unittest import TestCase, main as unittest_main, skipIf

from numpy import allclose, array, isclose, ndarray

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from torch import bfloat16, finfo
from torch.cuda import device_count

from tests.reimpl_t5.match_weights import *
from tests.reimpl_t5.experiment_mixins import *


def read_train_loss(path: str) -> ndarray:
    event_accumulator = EventAccumulator(path)
    event_accumulator.Reload()

    scalars = event_accumulator.Scalars("train/loss")

    return array([s.value for s in scalars])


def print_e2e_train_losses():
    reimpl_exp = TestBaselineExperiment()
    t5_exp = TestBaselineT5Experiment()

    reimpl_train_loss = read_train_loss(reimpl_exp.output_dir)
    t5_train_loss = read_train_loss(t5_exp.output_dir)
    diff = reimpl_train_loss - t5_train_loss

    atol = finfo(bfloat16).eps
    values_isclose = isclose(reimpl_train_loss, t5_train_loss, atol=atol)

    print(reimpl_train_loss, t5_train_loss, diff, values_isclose, sep="\n")


@skipIf(device_count() == 0, "Need GPUs to run end to end experiment")
class TestReimplMatchT5EndToEnd(TestCase):
    SEED = 42  # Only used for this test case

    def test_end_to_end_train(self) -> None:
        reimpl_exp = TestBaselineExperiment()
        if not exists(reimpl_exp.output_dir):
            reimpl_exp.train()

        t5_exp = TestBaselineT5Experiment()
        if not exists(t5_exp.output_dir):
            t5_exp.train()

        reimpl_train_loss = read_train_loss(reimpl_exp.output_dir)
        t5_train_loss = read_train_loss(t5_exp.output_dir)

        atol = finfo(bfloat16).eps
        self.assertTrue(allclose(reimpl_train_loss, t5_train_loss, atol=atol))


if __name__ == "__main__":
    unittest_args = argv[:1]
    unittest_main(argv=unittest_args)
