from march.experiments.baseline import *
from march.experiments.tests import *
from march.experiments.more_heads_less_layers_no_kv import *
from march.experiments.more_heads_less_layers import *
from march.experiments.more_dim_less_layers import *
from march.experiments.more_heads_more_dim_less_layers import *
from march.experiments.tp_embeddings_before_w_o import *

experiments_to_check = [
]

print("BaselineExperiment config params 222,054,912")
for experiment in experiments_to_check:
    model = experiment().get_model()
    print(f"{experiment.__name__} config params", f"{model.count_parameters():,}")

# config = TransformerConfig()
# model = BaselineTransformer(config)
# print("Baseline config params", f"{model.count_parameters():,}")
