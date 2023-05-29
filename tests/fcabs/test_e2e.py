from unittest import TestCase

from os.path import exists

from tests.fcabs.experiment_mixins import TestFCABSExperiment


class TestFCABSEndToEnd(TestCase):
    def test_end_to_end(self) -> None:
        exp = TestFCABSExperiment()
        if not exists(exp.output_dir): exp.train()
