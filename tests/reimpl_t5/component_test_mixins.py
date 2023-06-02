from torch import bfloat16, long, rand, randint
from torch.nn.functional import embedding

from tests.reimpl_t5.experiment_mixins import *


__all__ = ["ComponentTestMixin"]


class ComponentTestMixin:
    SEED = 42

    def setUp(self) -> None:
        # Initialize reimpl and t5 models
        reimpl_exp = TestBaselineExperiment()
        t5_exp = TestBaselineT5Experiment()
        reimpl_model = reimpl_exp.get_model()
        t5_model = t5_exp.get_model()
        reimpl_model = reimpl_model.to(bfloat16)
        t5_model = t5_model.to(bfloat16)

        # Reset parameters
        reimpl_exp._call_init_weights(reimpl_model, self.SEED)
        t5_exp._call_init_weights(t5_model, self.SEED)

        self.reimpl_exp = reimpl_exp
        self.reimpl_model = reimpl_model
        self.t5_exp = t5_exp
        self.t5_model = t5_model

        # Create dummy inputs
        N, L = 2, 8
        D = reimpl_model.config.dim_model
        self.input_ids = randint(0, reimpl_model.config.vocab_size, (N, L), dtype=long)
        self.reimpl_input_embeds = embedding(
            self.input_ids, reimpl_model.embedding.weight
        ).to(bfloat16)
        self.t5_input_embeds = t5_model.shared(self.input_ids).to(bfloat16)
        self.encoder_hidden_state = rand((N, L, D), dtype=bfloat16)

        self.attention_mask = randint(0, 2, (N, L), dtype=bool)
        self.t5_attention_mask = t5_model.get_extended_attention_mask(
            ~self.attention_mask, (N, L)
        )
        self.decoder_attention_mask = randint(0, 2, (N, L, L), dtype=bool)
        self.t5_decoder_attention_mask = t5_model.get_extended_attention_mask(
            ~self.decoder_attention_mask, (N, L, L)
        )
