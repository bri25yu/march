from typing import Dict, List, Optional

from os.path import join

import json

from datasets import Dataset, DatasetDict, load_dataset

from evaluate import load

from torch.utils.data import Sampler

from transformers import *
from transformers.integrations import TensorBoardCallback

from huggingface_hub import Repository, create_repo

from . import ROOT_DIR
from .common import *
from .dataset import IntToTextLabel, MetricForBestModel, currently_supported_tasks, pack_superglue


config_path = join(ROOT_DIR, "training_args.json")


class NoHParamsTensorBoardCallback(TensorBoardCallback):
    # This is a copy of `TensorBoardCallback.on_train_begin` unless specified otherwise
    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = None

        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = join(args.logging_dir, trial_name)

        if self.tb_writer is None:
            self._init_summary_writer(args, log_dir)

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    self.tb_writer.add_text("model_config", model_config_json)

            ###########################
            # START no hparams call
            ###########################

            # Original code:
            # # Version of TensorBoard coming from tensorboardX does not have this method.
            # if hasattr(self.tb_writer, "add_hparams"):
            #     self.tb_writer.add_hparams(args.to_sanitized_dict(), metric_dict={})

            ###########################
            # END no hparams call
            ###########################


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Callbacks
        for callback_to_remove in [PrinterCallback, ProgressCallback, TensorBoardCallback]:
            self.remove_callback(callback_to_remove)

        self.add_callback(NoHParamsTensorBoardCallback)

        # Compute metrics for multiple tasks at once
        self.current_eval_metric_key_prefix: Optional[str] = None

        self.compute_metrics = self.custom_compute_metrics

    # Fixes the order of samples seen
    def _get_train_sampler(self) -> Optional[Sampler]:
        return self._get_eval_sampler(self.train_dataset)

    def custom_compute_metrics(self, eval_preds: EvalPrediction) -> Dict[str, float]:
        task_name = self.current_eval_metric_key_prefix.removeprefix("eval_")
        tokenizer = self.tokenizer

        metric = load("super_glue", task_name)

        int_to_text_label = getattr(IntToTextLabel, task_name)
        text_to_int_label = {v: k for k, v in int_to_text_label.items()}

        prediction_ids, reference_ids = eval_preds.predictions, eval_preds.label_ids

        prediction_texts = tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
        reference_ids[reference_ids == -100] = tokenizer.pad_token_id  # Undo special label padding with -100
        reference_texts = tokenizer.batch_decode(reference_ids, skip_special_tokens=True)

        predictions = [text_to_int_label.get(s, -1) for s in prediction_texts]
        references = [text_to_int_label[s] for s in reference_texts]

        return metric.compute(predictions=predictions, references=references)

    # This is a copy of `Trainer._maybe_log_save_evaluate` unless specified otherwise
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            # if is_torch_tpu_available():
            #     xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)

                ##############################
                # START aggregate metric
                ##############################

                metrics["eval_aggregate"] = self.get_aggregate_metric(metrics)

                ##############################
                # END aggregate metric
                ##############################

            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)

            ##############################
            # START log metrics once
            ##############################

            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

            ##############################
            # END log metrics once
            ##############################

            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    # This is an exact copy of `Trainer.evaluate` unless specified otherwise
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """

        # Taken from `Seq2SeqTrainer.evaluate``
        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        ##############################
        # START set metric key prefix
        ##############################

        self.current_eval_metric_key_prefix = metric_key_prefix  # For self.compute_metrics

        ##############################
        # END set metric key prefix
        ##############################

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        # start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # total_batch_size = self.args.eval_batch_size * self.args.world_size
        # if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
        #     start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        # output.metrics.update(
        #     speed_metrics(
        #         metric_key_prefix,
        #         start_time,
        #         num_samples=output.num_samples,
        #         num_steps=math.ceil(output.num_samples / total_batch_size),
        #     )
        # )

        # Moving this call to _maybe_log_save_evaluate
        # self.log(output.metrics)

        # if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
        #     # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
        #     xm.master_print(met.metrics_report())

        # Moving this call to _maybe_log_save_evaluate
        # self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def get_aggregate_metric(self, metrics: Dict[str, float]) -> float:
        aggregate = 0.0  # Get mean of best metrics
        for task_name in currently_supported_tasks:
            best_metric_name = getattr(MetricForBestModel, task_name)
            best_metric_name = f"eval_{task_name}_{best_metric_name}"
            aggregate += metrics[best_metric_name]

        return aggregate / len(currently_supported_tasks)


def train(model_name: str, remote_dataset_path: Optional[str]=None) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_name_for_path = model_name.split("/")[-1]
    if remote_dataset_path is not None:
        superglue_packed_dataset_dict = load_dataset(remote_dataset_path)
    else:
        superglue_packed_dataset_dict = pack_superglue(tokenizer, model_name_for_path)

    hf_path = hf_path_template.format(model_name_for_path=model_name_for_path)
    local_dir = join(ROOT_DIR, "..", hf_path)
    repo = Repository(local_dir, clone_from=create_repo(hf_path, exist_ok=True))
    repo.git_pull()

    with open(config_path) as training_args_file:
        default_training_args_dict = json.load(training_args_file)

    eval_dataset_dict = DatasetDict({
        k.removeprefix("validation_"): v for k, v in superglue_packed_dataset_dict.items() if k.startswith("validation_")
    })

    for num_examples_per_input in ([1] + num_examples_per_input_list):
        output_dir = join(local_dir, f"{num_examples_per_input}packed")

        # This is where we adjust the batch size to hold the data budget constant
        assert examples_per_step % num_examples_per_input == 0
        batch_size = examples_per_step // num_examples_per_input

        print(f"For {num_examples_per_input} examples per input, using batch_size={batch_size}")

        args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=examples_per_step * 2,
            max_steps=max_steps,
            logging_dir=output_dir,
            metric_for_best_model="aggregate",
            **default_training_args_dict,
        )

        trainer = CustomSeq2SeqTrainer(
            model=AutoModelForSeq2SeqLM.from_pretrained(model_name),
            args=args,
            train_dataset=superglue_packed_dataset_dict[f"train_{num_examples_per_input}packed"],
            eval_dataset=eval_dataset_dict,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer),
        )

        trainer.train()

        for key in eval_dataset_dict.keys():
            trainer.current_eval_metric_key_prefix = f"eval_{key}"
            metrics = trainer.predict(eval_dataset_dict[key], metric_key_prefix=f"eval_final_{key}").metrics
            trainer.log(metrics)

        repo.push_to_hub(commit_message=f"Adding {num_examples_per_input} packed")
