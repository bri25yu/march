from torch import randint

from march.experiments.baseline import *
from march.experiments.tests import *


experiments_to_check = []


for experiment_cls in experiments_to_check:
    experiment = experiment_cls()
    model = experiment.get_model()

    N, L = 2, 128
    input_ids = randint(0, model.config.vocab_size, (N, L))
    attention_mask = randint(0, 2, (N, L))

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=input_ids,
        decoder_attention_mask=attention_mask[:, None, :],
        labels=input_ids,
    )
    outputs.logits.mean().backward()

    print(outputs.logits[0, 0, :10])
    print(f"{experiment.name} passed!")
