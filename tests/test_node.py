import march  # Redirect cache. This is necessary because this file might be run as a standalone

from sys import argv

from unittest import TestCase, main as unittest_main, skipIf

from torch.cuda import device_count

from tests.reimpl_t5.experiment_mixins import *


@skipIf(device_count() == 0, "Need GPUs to run node test")
class TestNode(TestCase):
    def test_node_basic(self) -> None:
        exp = TestBaselineT5Experiment()
        exp.train()


if __name__ == "__main__":
    unittest_args = argv[:1]
    unittest_main(argv=unittest_args)
