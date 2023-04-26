from march.experiments.tests import *


experiments_to_run = [
]

print("BaselineExperiment config params 222,054,912")
for experiment in experiments_to_run:
    model = experiment().get_model()
    print(f"{experiment.__name__} config params", f"{model.count_parameters():,}")

# config = TransformerConfig()
# model = BaselineTransformer(config)
# print("Baseline config params", f"{model.count_parameters():,}")
