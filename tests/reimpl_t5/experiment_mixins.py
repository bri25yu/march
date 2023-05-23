from march.experiments.baseline import BaselineSmallFullTrainExperiment, BaselineT5SmallExperiment


__all__ = ["TestBaselineExperiment", "TestBaselineT5Experiment"]


class TestExperimentMixin:
    NUM_STEPS = 100


class TestBaselineExperiment(TestExperimentMixin, BaselineSmallFullTrainExperiment):
    pass


class TestBaselineT5Experiment(TestExperimentMixin, BaselineT5SmallExperiment):
    pass
