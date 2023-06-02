from transformers import Seq2SeqTrainingArguments

from march.models.baseline import TransformerBase
from march.models.fixed_change_attention_based_summarization import (
    FCABSTransformerConfig,
    FCABSTransformer,
)

from march.experiments.tests import FCABSExperiment


__all__ = ["TestFCABSExperiment"]


class TestFCABSExperiment(FCABSExperiment):
    NUM_STEPS = 10

    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        args_dict = self.load_default_training_arguments()

        args_dict["eval_steps"] = 1
        args_dict["eval_accumulation_steps"] = 1

        return Seq2SeqTrainingArguments(**args_dict)

    def get_model(self) -> TransformerBase:
        config = FCABSTransformerConfig(
            dim_model=512,
            num_layers=12,
        )
        return FCABSTransformer(config)
