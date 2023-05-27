from typing import Tuple

from numpy import array, ndarray

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from transformers import AutoTokenizer


def get_dropped_ids(path: str) -> Tuple[ndarray, ndarray]:
    event_accumulator = EventAccumulator(path)
    event_accumulator.Reload()

    texts = event_accumulator.Tensors("eval/global_dropped_ids_by_layer/text_summary")

    steps = array([s.step for s in texts])
    ids = array([eval(s.tensor_proto.string_val[0]) for s in texts])

    return steps, ids


# Values is (eval_steps, num_layers, batch_size, L_drop)
steps, ids = get_dropped_ids("results/FCABSLdrop32Experiment")

ids_ex = ids[-1, :, 0, :]  # Most recent eval step with first example
num_layers, L_drop = ids_ex.shape
ids_ex = ids_ex.reshape(-1)[:, None]

tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=1024)
tokens_ex = tokenizer.batch_decode(ids_ex)
tokens_ex = array(tokens_ex).reshape(num_layers, L_drop)

for i, tokens in enumerate(tokens_ex):
    print(f"Layer {i} dropped ids", tokens, "", sep="\n")
