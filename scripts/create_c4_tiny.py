import march  # Redirect cache

from march.datasets.c4 import load_c4


c4_10m = load_c4()
c4_100 = c4_10m["train"].select(range(100)).push_to_hub("c4_t5_100")
