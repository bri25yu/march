from transformers import PreTrainedTokenizerFast, Seq2SeqTrainingArguments

from march.experiments.baseline import BaselineExperiment


class TestBaselineRuntimeExperiment(BaselineExperiment):
    NUM_STEPS = 15

    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        return Seq2SeqTrainingArguments(**default_training_arguments)


class TestBaselineRuntimeShorterExperiment(TestBaselineRuntimeExperiment):
    """
    Max seq len from 1024 -> 256 (labels from 300 -> 80)
    Per GPU batch size from 8 -> 32 to compensate for max seq len reduction
    Per GPU batch size from 32 -> 16 to have half the data budget
    Total change is per GPU batch size is a factor of 2x
    """

    MAX_INPUT_IDS_LEN = 256
    MAX_LABELS_LEN = 80

    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        args_dict = self.load_default_training_arguments()

        args_dict["per_device_train_batch_size"] = (
            args_dict["per_device_train_batch_size"] * 2
        )
        args_dict["per_device_eval_batch_size"] = (
            args_dict["per_device_eval_batch_size"] * 4
        )

        return Seq2SeqTrainingArguments(**args_dict)

    def get_data_collator(self, tokenizer: PreTrainedTokenizerFast):
        data_collator = super().get_data_collator(tokenizer)

        def shortened_seq_len_data_collator(examples):
            for example in examples:
                example["input_ids"] = example["input_ids"][: self.MAX_INPUT_IDS_LEN]
                example["labels"] = example["labels"][: self.MAX_LABELS_LEN]

            examples = data_collator(examples)

            return examples

        return shortened_seq_len_data_collator
