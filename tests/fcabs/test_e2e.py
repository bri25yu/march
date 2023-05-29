import march  # Redirect cache. This is necessary because this file might be run as a standalone

from sys import argv

from os.path import exists

from unittest import TestCase, main as unittest_main

from tests.fcabs.experiment_mixins import TestFCABSExperiment


class TestFCABSEndToEnd(TestCase):
    def test_end_to_end(self) -> None:
        exp = TestFCABSExperiment(batch_size_pow_scale=-1)
        if not exists(exp.output_dir): exp.train()


if __name__ == "__main__":
    unittest_args = argv[:1]
    unittest_main(argv=unittest_args)
