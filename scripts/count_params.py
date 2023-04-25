from march.models.big_heads import BigHeadsTransformer, BigHeadsTransformerConfig
from march.models.baseline import BaselineTransformer, TransformerConfig 
from march.experiments.tests import *
from march.experiments.baseline import *


experiments_to_run = [
    BaselineFP32Experiment,
    ReLUGatedLinearUnitExperiment,
    GELUGatedLinearUnitExperiment,
    SiLUGatedLinearUnitExperiment,
    MixedActExperiment,
    MixedActSumOverMeanExperiment,
    MixedActSOMDropoutExperiment,
    NoSelfAttentionResidualExperiment,
    MoreHeadsLessQKVDimExperiment,
    MoreHeadsLessLayersExperiment,
    LessHeadsMoreQKVDimExperiment,
    MoreHeadsLessQKVDimLessLayersExperiment,
    APESumOverAverageExperiment,
    APEUnitVarianceExperiment
]

for experiment in experiments_to_run:
    model = experiment().get_model()
    print(f"{experiment.__name__} config params", f"{model.count_parameters():,}")

config = TransformerConfig()
model = BaselineTransformer(config)
print("Baseline config params", f"{model.count_parameters():,}")


# model = BigHeadsDownProjectExperiment().get_model()
# print(f"{model.count_parameters():,}")


# model = BigHeadsDownProject2Experiment().get_model()
# print(f"{model.count_parameters():,}")


# model = BigHeadsLinearW_o3Experiment().get_model()
# print(f"{model.count_parameters():,}")

