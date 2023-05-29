from os.path import join

import json

from torch.cuda import device_count

from transformers import Seq2SeqTrainingArguments

from march import CONFIG_DIR
from march.experiments.tests import FCABSExperiment


__all__ = ["TestFCABSExperiment", "TestFCABSCPUExperiment"]


class TestFCABSExperiment(FCABSExperiment):
    NUM_STEPS = 10

    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        args_dict = self.load_default_training_arguments()
        args_dict["eval_steps"] = 1
        return Seq2SeqTrainingArguments(**args_dict)


class TestFCABSCPUExperiment(FCABSExperiment):
    NUM_STEPS = 10

    # Most of this is copied from ExperimentBase
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        assert device_count() == 0

        with open(join(CONFIG_DIR, "default_training_arguments.json")) as default_training_args_file:
            args_dict = json.load(default_training_args_file)

        args_dict["eval_steps"] = 1

        args_dict["per_device_train_batch_size"] = 1
        args_dict["per_device_eval_batch_size"] = 1
        args_dict["gradient_accumulation_steps"] = 1

        # The rest of this function is from `load_default_training_arguments`
        if self.NUM_STEPS is not None:
            assert isinstance(self.NUM_STEPS, int)
            args_dict["max_steps"] = self.NUM_STEPS

        # Output and logging directories
        args_dict["output_dir"] = self.output_dir
        args_dict["logging_dir"] = self.output_dir

        return Seq2SeqTrainingArguments(**args_dict)
