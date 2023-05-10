from march.models.TPEmbeddings_before_w_o import *

from march.experiments.baseline import *


class TPEmbeddingsBeforeW_oEncDecSharedExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TPEmbeddingsBeforeW_oConfig(num_layers=24,num_roles=2000)
        return TPEmbeddingsBeforeW_oEncDecSharedTransformer(config)


class TPEmbeddingsBeforeW_oEncDecShared2Experiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_half_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)

    def get_model(self) -> TransformerBase:
        config = TPEmbeddingsBeforeW_oConfig(num_layers=24,num_roles=10000)
        return TPEmbeddingsBeforeW_oEncDecSharedTransformer(config)


class TPEmbeddingsBeforeW_oEncDecSharedCross2Experiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_half_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)
        
    def get_model(self) -> TransformerBase:
        config = TPEmbeddingsBeforeW_oConfig(num_layers=24,num_roles=10000,role_attention_type=1)
        return TPEmbeddingsBeforeW_oEncDecSharedTransformer(config)


class TPEmbeddingsBeforeW_oEncDecSharedBoth2Experiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_half_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)
        
    def get_model(self) -> TransformerBase:
        config = TPEmbeddingsBeforeW_oConfig(num_layers=24,num_roles=10000,role_attention_type=2)
        return TPEmbeddingsBeforeW_oEncDecSharedTransformer(config)


class TPEmbeddingsBeforeW_oEncDecNotSharedExperiment(BaselineExperiment):
    def get_model(self) -> TransformerBase:
        config = TPEmbeddingsBeforeW_oConfig(num_layers=24,num_roles=2000)
        return TPEmbeddingsBeforeW_oEncDecNotSharedTransformer(config)


class TPEmbeddingsBeforeW_oEncDecNotShared2Experiment(BaselineExperiment):
    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        default_training_arguments = self.load_default_training_arguments()
        default_training_arguments = update_with_half_batch_size(default_training_arguments)

        return Seq2SeqTrainingArguments(self.output_dir, **default_training_arguments)
        
    def get_model(self) -> TransformerBase:
        config = TPEmbeddingsBeforeW_oConfig(num_layers=24,num_roles=10000)
        return TPEmbeddingsBeforeW_oEncDecNotSharedTransformer(config)
