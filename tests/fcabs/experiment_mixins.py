from transformers import Seq2SeqTrainingArguments
from march.experiments.tests import FCABSExperiment


__all__ = ["TestFCABSExperiment"]


class TestFCABSExperiment(FCABSExperiment):
    NUM_STEPS = 10

    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        args_dict = self.load_default_training_arguments()
        args_dict["eval_steps"] = 1
        return Seq2SeqTrainingArguments(**args_dict)
