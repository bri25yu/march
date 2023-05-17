from torch import equal, long, manual_seed as set_torch_seed, randint, ones

from march.models.baseline import *
from march.models.utils import *

from march.experiments.baseline import BaselineExperiment, BaselineT5Experiment, ExperimentBase


SEED = 42

print("Initializing reimpl and t5 models")
reimpl_exp = BaselineExperiment()
t5_exp = BaselineT5Experiment()
reimpl_model = reimpl_exp.get_model()
t5_model = t5_exp.get_model()

print("Resetting parameters")
reimpl_exp._call_init_weights(reimpl_model, SEED)
t5_exp._call_init_weights(t5_model, SEED)

# Sanity check weight matching
reimpl_weight = reimpl_model.encoder.self_attention_layers[0].w_q
t5_weight = t5_model.encoder.block[0].layer[0].SelfAttention.q
reimpl_weight = reimpl_weight.weight.data
t5_weight = t5_weight.weight.data

assert equal(reimpl_weight, t5_weight)

N, L = 2, 128
input_ids = randint(0, reimpl_model.config.vocab_size, (N, L), dtype=long)
input_embeds = t5_model.shared(input_ids)
attention_mask = ones((N, L), dtype=long)

# Encoder
set_torch_seed(SEED)
reimpl_encoder_outputs = reimpl_model.encoder(
    input_embeds=input_embeds,
    attention_mask=attention_mask,
).input_embeds

set_torch_seed(SEED)
t5_encoder_outputs = t5_model.encoder(
    inputs_embeds=input_embeds,
    attention_mask=1.0 - attention_mask,
)[0]

assert equal(reimpl_encoder_outputs, t5_encoder_outputs)

# Decoder
decoder_attention_mask = reimpl_model.create_decoder_attention_mask(input_ids)
set_torch_seed(SEED)
reimpl_decoder_outputs = reimpl_model.decoder(
    input_embeds=input_embeds,
    attention_mask=decoder_attention_mask,
    encoder_hidden_state=reimpl_encoder_outputs,
    encoder_attention_mask=attention_mask,
).input_embeds
set_torch_seed(SEED)
t5_decoder_outputs = t5_model.decoder(
    inputs_embeds=input_embeds,
    attention_mask=1.0 - decoder_attention_mask,
    encoder_hidden_states=t5_encoder_outputs,
    encoder_attention_mask=1.0 - attention_mask,
)[0]

assert equal(reimpl_decoder_outputs, t5_decoder_outputs)

set_torch_seed(SEED)
reimpl_outputs = reimpl_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    decoder_input_ids=input_ids,
    labels=input_ids,
)
set_torch_seed(SEED)
t5_outputs = t5_model(
    input_ids=input_ids,
    attention_mask=1.0 - attention_mask,
    decoder_input_ids=input_ids,
    labels=input_ids,
)

reimpl_logits = reimpl_outputs.logits
t5_logits = t5_outputs.logits
assert equal(reimpl_logits, t5_logits)

reimpl_loss = reimpl_outputs.loss
t5_loss = t5_outputs.loss
assert equal(reimpl_loss, t5_loss)

# Check gradients
reimpl_loss.backward()
t5_loss.backward()

reimpl_weight = reimpl_model.encoder.self_attention_layers[0].w_q
t5_weight = t5_model.encoder.block[0].layer[0].SelfAttention.q
reimpl_grad = reimpl_weight.weight.grad
t5_grad = t5_weight.weight.grad
assert equal(reimpl_grad, t5_grad)
