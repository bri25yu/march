from march.models.big_heads import BigHeadsTransformer, BigHeadsTransformerConfig
from march.models.baseline import BaselineTransformer, TransformerConfig 
from march.experiments.tests import *

config = TransformerConfig()
model = BaselineTransformer(config)
print("Baseline config params", f"{model.count_parameters():,}")


model = BigHeadsDownProjectExperiment().get_model()
print(f"{model.count_parameters():,}")


model = BigHeadsDownProject2Experiment().get_model()
print(f"{model.count_parameters():,}")


# model = BigHeadsLinearW_o3Experiment().get_model()
# print(f"{model.count_parameters():,}")

