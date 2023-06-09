from march.models.utils import count_parameters

from march.experiments.baseline import *
from march.experiments.tests import *


experiments_to_check = []

print("BaselineExperiment config params 222,903,552")
for experiment in experiments_to_check:
    model = experiment().get_model()
    print(f"{experiment.__name__} config params", f"{count_parameters(model):,}")
