from os.path import exists

from unittest import TestCase

from tests.fcabs.experiment_mixins import TestFCABSCPUExperiment


class TestFCABSIntegration(TestCase):
    def test_cpu(self) -> None:
        exp = TestFCABSCPUExperiment()
        if not exists(exp.output_dir): exp.train()
