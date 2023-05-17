from torch import equal, long, manual_seed as set_torch_seed, rand, randint, set_grad_enabled

from transformers.models.t5.modeling_t5 import T5Attention

from march.models.baseline import BaselineAttention

from march.experiments.baseline import BaselineExperiment, BaselineT5Experiment


set_grad_enabled(False)

SEED = 42

print("Initializing reimpl and t5 models")
reimpl_model = BaselineExperiment().get_model()
t5_model = BaselineT5Experiment().get_model()

reimpl_attn = BaselineAttention(reimpl_model.config, is_cross_attention=True, has_relative_attention_bias=False)
t5_attn = T5Attention(t5_model.config, has_relative_attention_bias=False)
t5_attn.is_decoder = True

set_torch_seed(SEED)
reimpl_attn._init_weights()
set_torch_seed(SEED)
t5_model._init_weights(t5_attn)

# Test equality
reimpl_attn.eval()
t5_attn.eval()

N, L = 2, 128
D = reimpl_attn.config.dim_model

input_ids = randint(0, reimpl_model.config.vocab_size, (N, L), dtype=long)
input_embeds = t5_model.shared(input_ids)
encoder_hidden_state = rand((N, L, D))

attention_mask = reimpl_model.create_decoder_attention_mask(input_ids)
t5_attn_mask = t5_model.get_extended_attention_mask(1.0 - attention_mask, (N, L))

reimpl_outputs = reimpl_attn(input_embeds, attention_mask, encoder_hidden_state=encoder_hidden_state).input_embeds
t5_outputs = t5_attn(input_embeds, t5_attn_mask, key_value_states=encoder_hidden_state)[0]
assert equal(reimpl_outputs, t5_outputs)
