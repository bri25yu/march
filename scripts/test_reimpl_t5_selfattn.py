from torch import equal, long, manual_seed as set_torch_seed, ones, randint

from transformers.models.t5.modeling_t5 import T5Attention

from march.models.baseline import BaselineAttention

from march.experiments.baseline import BaselineExperiment, BaselineT5Experiment


SEED = 42

print("Initializing reimpl and t5 models")
reimpl_model = BaselineExperiment().get_model()
t5_model = BaselineT5Experiment().get_model()

reimpl_attn = BaselineAttention(reimpl_model.config, is_cross_attention=False, has_relative_attention_bias=True)
t5_attn = T5Attention(t5_model.config, has_relative_attention_bias=True)

set_torch_seed(SEED)
reimpl_attn._init_weights()
set_torch_seed(SEED)
t5_model._init_weights(t5_attn)

# Test equality
N, L = 2, 128
input_ids = randint(0, reimpl_model.config.vocab_size, (N, L), dtype=long)
input_embeds = t5_model.shared(input_ids)
attention_mask = ones((N, L), dtype=long)
t5_attn_mask = t5_model.get_extended_attention_mask(1.0 - attention_mask, (N, L))

set_torch_seed(SEED)
reimpl_outputs = reimpl_attn(input_embeds, attention_mask).input_embeds
set_torch_seed(SEED)
t5_outputs = t5_attn(input_embeds, t5_attn_mask)[0]


assert equal(reimpl_outputs, t5_outputs)
