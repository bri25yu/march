from transformers.trainer_utils import EvalPrediction
from transformers import Seq2SeqTrainingArguments

from march.models.baseline import TransformerBase, TransformerConfig
from march.models.no_ff import NoFFTransformer
from march.models.values_relu import ValuesReluTransformer, ValuesReluFirstFFTransformer
from march.models.fixed_change_attention_based_summarization import (
    FCABSTransformer,
    FCABSTransformerConfig,
)

from march.experiments.baseline import BaselineExperiment, BaselineT5Experiment


class NoFFExperiment(BaselineExperiment):
    NUM_STEPS = 1_000

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return NoFFTransformer(config)


class NoFFParamMatchExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=1088)
        return NoFFTransformer(config)


class ValuesReluExperiment(BaselineExperiment):
    NUM_STEPS = 1_000

    def get_model(self) -> TransformerBase:
        config = TransformerConfig()
        return ValuesReluTransformer(config)


class ValuesReluNoUpProjExperiment(BaselineExperiment):
    NUM_STEPS = 1_000

    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=960, feedforward_scale=1)
        return ValuesReluTransformer(config)


class ValuesReluFirstFFExperiment(BaselineExperiment):
    NUM_STEPS = 1_000

    def get_model(self) -> TransformerBase:
        config = TransformerConfig(dim_model=1024)
        return ValuesReluFirstFFTransformer(config)


class FCABSExperiment(BaselineExperiment):
    NUM_STEPS = 1_000

    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        args_dict = self.load_default_training_arguments()

        # We use eval accumulation steps value of 1 because the logging intermediate
        # values are very large
        args_dict["eval_accumulation_steps"] = 1

        return Seq2SeqTrainingArguments(**args_dict)

    def get_model(self) -> TransformerBase:
        config = FCABSTransformerConfig()
        return FCABSTransformer(config)

    def get_compute_metrics(self, model: TransformerBase):
        def compute_metrics(eval_preds: EvalPrediction):
            logs = dict(model.named_parameters())

            logits, dropped_ids = eval_preds.predictions

            return {**logs, "dropped_ids_by_layer": dropped_ids}

        return compute_metrics


class FCABSLdrop32Experiment(FCABSExperiment):
    NUM_STEPS = 10_000

    def get_model(self) -> TransformerBase:
        config = FCABSTransformerConfig(L_drop=32)
        return FCABSTransformer(config)


class TestA25Experiment(BaselineT5Experiment):
    pass
