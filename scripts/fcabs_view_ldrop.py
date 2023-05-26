from typing import List, Tuple

from numpy import array

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_dropped_ids(path: str) -> Tuple[List[float], List[float]]:
    event_accumulator = EventAccumulator(path)
    event_accumulator.Reload()

    texts = event_accumulator.Tensors("eval/global_dropped_ids_by_layer")

    steps = array([s.step for s in texts])
    values = array([s.value for s in texts])

    return steps, values


steps, values = get_dropped_ids("results/FCABSLdrop32Experiment")
print(values[0])
