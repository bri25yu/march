from tqdm.auto import trange

from numpy.random import seed as set_numpy_seed

from torch import equal, long, manual_seed as set_torch_seed, no_grad, randint, ones

from march.models.baseline import *
from march.models.utils import *

from march.experiments.baseline import BaselineExperiment, BaselineT5Experiment


SEED = 42
set_numpy_seed(SEED)
set_torch_seed(SEED)

print("Initializing baseline and t5 models")
baseline_model = BaselineExperiment().get_model()
baseline_t5_model = BaselineT5Experiment().get_model()


# Match weight values. This won't check for correctness for weight init, but that's ok
print("Matching params")
total_matched_params = 0

with no_grad():
    baseline_model.embedding.weight.copy_(baseline_t5_model.shared.weight)
    total_matched_params += baseline_model.embedding.weight.numel()

    to_match = [
        (baseline_model.encoder, baseline_t5_model.encoder, False),
        (baseline_model.decoder, baseline_t5_model.decoder, True),
    ]

    for reimpl, t5, match_crossattn in to_match:
        for i in range(baseline_model.config.num_layers // 2):
            baseline_selfattn = reimpl.self_attention_layers[i]
            t5_selfattn = t5.block[i].layer[0].SelfAttention

            baseline_selfattn.w_q.weight.copy_(t5_selfattn.q.weight)
            baseline_selfattn.w_k.weight.copy_(t5_selfattn.k.weight)
            baseline_selfattn.w_v.weight.copy_(t5_selfattn.v.weight)
            baseline_selfattn.w_o.weight.copy_(t5_selfattn.o.weight)

            total_matched_params += baseline_selfattn.w_q.weight.numel()
            total_matched_params += baseline_selfattn.w_k.weight.numel()
            total_matched_params += baseline_selfattn.w_v.weight.numel()
            total_matched_params += baseline_selfattn.w_o.weight.numel()

            # Match selfattn position bias
            if i == 0:
                baseline_selfattn.relative_attention_bias.weight.copy_(t5_selfattn.relative_attention_bias.weight)
                total_matched_params += baseline_selfattn.relative_attention_bias.weight.numel()

            # Match selfattn layernorm
            baseline_selfattn_layernorm = reimpl.layernorms[2 * i]
            t5_selfattn_layernorm = t5.block[i].layer[0].layer_norm
            baseline_selfattn_layernorm.weight.copy_(t5_selfattn_layernorm.weight)
            total_matched_params += baseline_selfattn_layernorm.weight.numel()

            # Match cross attn
            if match_crossattn:
                baseline_crossattn = reimpl.cross_attention_layers[i]
                t5_crossattn = t5.block[i].layer[1].EncDecAttention

                baseline_crossattn.w_q.weight.copy_(t5_crossattn.q.weight)
                baseline_crossattn.w_k.weight.copy_(t5_crossattn.k.weight)
                baseline_crossattn.w_v.weight.copy_(t5_crossattn.v.weight)
                baseline_crossattn.w_o.weight.copy_(t5_crossattn.o.weight)

                total_matched_params += baseline_crossattn.w_q.weight.numel()
                total_matched_params += baseline_crossattn.w_k.weight.numel()
                total_matched_params += baseline_crossattn.w_v.weight.numel()
                total_matched_params += baseline_crossattn.w_o.weight.numel()

                # Match crossattn layernorm
                baseline_crossattn_layernorm = reimpl.layernorms[2 * i + 1]
                t5_crossattn_layernorm = t5.block[i].layer[1].layer_norm
                baseline_crossattn_layernorm.weight.copy_(t5_crossattn_layernorm.weight)
                total_matched_params += baseline_crossattn_layernorm.weight.numel()

            # Match ff weights
            baseline_ff = reimpl.feedforward_layers[i]
            t5_ff = t5.block[i].layer[-1]

            baseline_ff.up_projection.weight.copy_(t5_ff.DenseReluDense.wi.weight)
            baseline_ff.down_projection.weight.copy_(t5_ff.DenseReluDense.wo.weight)

            total_matched_params += baseline_ff.up_projection.weight.numel()
            total_matched_params += baseline_ff.down_projection.weight.numel()

            # Match ff layernorm
            baseline_ff_layernorm = reimpl.layernorms[2 * i + 1 + (match_crossattn)]
            t5_ff_layernorm = t5_ff.layer_norm
            baseline_ff_layernorm.weight.copy_(t5_ff_layernorm.weight)
            total_matched_params += baseline_ff_layernorm.weight.numel()

        # Match encoder/decoder final layernorm
        baseline_final_layernorm = reimpl.layernorms[-1]
        t5_final_layernorm = t5.final_layer_norm
        baseline_final_layernorm.weight.copy_(t5_final_layernorm.weight)
        total_matched_params += baseline_final_layernorm.weight.numel()


assert total_matched_params == baseline_model.count_parameters(), (f"{total_matched_params:,}", f"{baseline_model.count_parameters():,}")

baseline_model.eval()
baseline_t5_model.eval()


with no_grad():
    for _ in trange(5, desc="Testing equality"):
        N, L = 2, 128
        input_ids = randint(0, baseline_model.config.vocab_size, (N, L), dtype=long)
        attention_mask = ones((N, L), dtype=long)

        baseline_outputs = baseline_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=input_ids,
            labels=input_ids,
        ).logits

        baseline_t5_outputs = baseline_t5_model(
            input_ids=input_ids,
            attention_mask=1.0 - attention_mask,
            decoder_input_ids=input_ids,
            labels=input_ids,
        ).logits

        # TODO Debug small diff
        print(baseline_outputs.size(), baseline_t5_outputs.size())
        if not equal(baseline_outputs, baseline_t5_outputs):
            print()
            print("reimpl outputs\n", baseline_outputs[0, -1, :10])
            print("t5 outputs\n", baseline_t5_outputs[0, -1, :10])
            print("diff\n", baseline_outputs[0, -1, :10] - baseline_t5_outputs[0, -1, :10])
            assert False
