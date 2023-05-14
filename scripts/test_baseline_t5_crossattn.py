from numpy.random import seed as set_numpy_seed

from torch import equal, long, manual_seed as set_torch_seed, ones, rand, randint, set_grad_enabled

from transformers.models.t5.modeling_t5 import T5Attention

from march.models.baseline import BaselineAttention

from march.experiments.baseline import BaselineExperiment, BaselineT5Experiment


SEED = 42
set_numpy_seed(SEED)
set_torch_seed(SEED)

set_grad_enabled(False)

print("Initializing reimpl and t5 models")
reimpl_model = BaselineExperiment().get_model()
t5_model = BaselineT5Experiment().get_model()


reimpl_attn = BaselineAttention(reimpl_model.config, is_cross_attention=True, has_relative_attention_bias=False)
t5_attn = T5Attention(t5_model.config, has_relative_attention_bias=False)
t5_attn.is_decoder = True

print("Matching params")
total_matched_params = 0

reimpl_attn.w_q.weight.copy_(t5_attn.q.weight)
reimpl_attn.w_k.weight.copy_(t5_attn.k.weight)
reimpl_attn.w_v.weight.copy_(t5_attn.v.weight)
reimpl_attn.w_o.weight.copy_(t5_attn.o.weight)

total_matched_params += reimpl_attn.w_q.weight.numel()
total_matched_params += reimpl_attn.w_k.weight.numel()
total_matched_params += reimpl_attn.w_v.weight.numel()
total_matched_params += reimpl_attn.w_o.weight.numel()

target_matched_params = reimpl_model.__class__.count_parameters(reimpl_attn)
assert total_matched_params == target_matched_params, (f"{total_matched_params:,}", f"{target_matched_params:,}")


# Test equality
reimpl_attn.eval()
t5_attn.eval()

N, L = 2, 128
D = reimpl_attn.config.dim_model

input_ids = randint(0, reimpl_model.config.vocab_size, (N, L), dtype=long)
input_embeds = t5_model.shared(input_ids)

attention_mask = ones((N, L), dtype=long)
t5_attn_mask = t5_model.get_extended_attention_mask(1.0 - attention_mask, (N, L))

encoder_hidden_state = rand((N, L, D))

reimpl_outputs = reimpl_attn(input_embeds, attention_mask, encoder_hidden_state=encoder_hidden_state).input_embeds
t5_outputs = t5_attn(input_embeds, t5_attn_mask, key_value_states=encoder_hidden_state)[0]


if not equal(reimpl_outputs, t5_outputs):
    print()
    print("reimpl outputs\n", reimpl_outputs[0, 0, :10])
    print("t5 outputs\n", t5_outputs[0, 0, :10])
    print("diff\n", reimpl_outputs[0, 0, :10] - t5_outputs[0, 0, :10])
    assert False
