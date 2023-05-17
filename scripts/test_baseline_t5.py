from torch import equal, long, manual_seed as set_torch_seed, randint, set_grad_enabled, ones

from march.models.baseline import *
from march.models.utils import *

from march.experiments.baseline import BaselineExperiment, BaselineT5Experiment, ExperimentBase


set_grad_enabled(False)

SEED = 42

print("Initializing baseline and t5 models")
baseline_model = BaselineExperiment().get_model()
t5_model = BaselineT5Experiment().get_model()

print("Resetting parameters")
set_torch_seed(SEED)
ExperimentBase._call_init_weights(None, baseline_model)
set_torch_seed(SEED)
t5_model.apply(t5_model._init_weights)

# Sanity check weight matching
reimpl_weight = baseline_model.encoder.self_attention_layers[0].w_q
t5_weight = t5_model.encoder.block[0].layer[0].SelfAttention.q
reimpl_weight = reimpl_weight.weight.data
t5_weight = t5_weight.weight.data

assert equal(reimpl_weight, t5_weight)

baseline_model.eval()
t5_model.eval()


N, L = 2, 128
input_ids = randint(0, baseline_model.config.vocab_size, (N, L), dtype=long)
input_embeds = t5_model.shared(input_ids)
attention_mask = ones((N, L), dtype=long)

# Encoder
reimpl_encoder_outputs = baseline_model.encoder(
    input_embeds=input_embeds,
    attention_mask=attention_mask,
).input_embeds

t5_encoder_outputs = t5_model.encoder(
    inputs_embeds=input_embeds,
    attention_mask=1.0 - attention_mask,
)[0]

assert equal(reimpl_encoder_outputs, t5_encoder_outputs)

# Decoder
decoder_attention_mask = baseline_model.create_decoder_attention_mask(input_ids)
reimpl_decoder_outputs = baseline_model.decoder(
    input_embeds=input_embeds,
    attention_mask=decoder_attention_mask,
    encoder_hidden_state=reimpl_encoder_outputs,
    encoder_attention_mask=attention_mask,
).input_embeds
t5_decoder_outputs = t5_model.decoder(
    inputs_embeds=input_embeds,
    attention_mask=1.0 - decoder_attention_mask,
    encoder_hidden_states=t5_encoder_outputs,
    encoder_attention_mask=1.0 - attention_mask,
)[0]

assert equal(reimpl_decoder_outputs, t5_decoder_outputs)

reimpl_outputs = baseline_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    decoder_input_ids=input_ids,
    labels=input_ids,
)
t5_outputs = t5_model(
    input_ids=input_ids,
    attention_mask=1.0 - attention_mask,
    decoder_input_ids=input_ids,
    labels=input_ids,
    decoder_attention_mask=1.0 - decoder_attention_mask,
)

reimpl_logits = reimpl_outputs.logits
t5_logits = t5_outputs.logits
assert equal(reimpl_logits, t5_logits)
