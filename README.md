# march
Model ARCHitecture experiments

Example setup:
```bash
git clone https://github.com/bri25yu/march
cd march

conda env create --file environment.yml
conda activate march

deepspeed run.py
```

# Experimental Setup
All of the following experiments are over constant data budget, model parameters, and compute unless noted otherwise. The data budget is determined by the number of steps taken and the number of tokens per step, for a total number of tokens seen over training. The model parameters is determined by counting the total number of trainable parameters in a model prior to training. The compute is approximated by how long the run took. All experiments are run on an NVIDIA A100 40GB GPU on Google Colab. 

We train models for 1000 steps, enough for the models to start learning and to make their behavior/performance differentiable from other models. Every step, the model sees 1M tokens, Every experiment sees 1000 steps * 1M tokens per step = 1B tokens. The baseline model has around 36M parameters and by default every subsequent model matches this budget. 


# Results
More heads better, GLU better

## MoreHeadsLessLayersExperiment
Num layers from 6 to 4, num heads from 8 to 16

Slightly better, slightly faster, slightly more unstable (?)

## LessHeadsMoreQKVDimExperiment
Dim qkv from 64 to 128, num heads from 8 to 4

Slightly worse, faster, maybe because there's like a minimum number of heads, then past that the model doesn't care.

## Scaling heads
[in progress] but the initial scaling heads is identical to the baseline

## APESumOverAverageExperiment / APEUnitVarianceExperiment
Different recombinations of input and output for position encoding. Generally worse than the baseline

## GatedLinearUnit experiments
All slightly better than the baseline and slightly more expensive.

# Mixed Act
All slightly worse than the baseline, slightly more varied. 

# No Self-Attention residual
Makes training significantly worse and more varied, maybe we shouldn't sparsify attention with no residual
