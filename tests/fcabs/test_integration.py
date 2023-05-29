from os import environ
environ["CUDA_VISIBLE_DEVICES"] = ""

import march  # Redirect cache. This is necessary because this file might be run as a standalone

from sys import argv

from os.path import exists

from unittest import TestCase, main as unittest_main

from unittest import TestCase

from tests.fcabs.experiment_mixins import TestFCABSCPUExperiment


class TestFCABSIntegration(TestCase):
    def test_cpu(self) -> None:
        exp = TestFCABSCPUExperiment()
        if not exists(exp.output_dir): exp.train()


if __name__ == "__main__":
    unittest_args = argv[:1]
    unittest_main(argv=unittest_args)
