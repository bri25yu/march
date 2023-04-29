# march
Model ARCHitecture experiments

Research conducted under Prof. Kurt Keutzer at Berkeley Artificial Intelligence Research (BAIR). 

<img src="http://bair.berkeley.edu/images/BAIR_Logo_BlueType_Tag.png" width="525" height="280">

Example setup:
```bash
git clone https://github.com/bri25yu/march
cd march

conda env create --file environment.yml
conda activate march

deepspeed run.py
```

# Experimental Setup
All of the following experiments are over constant data budget, model parameters, and compute unless noted otherwise. The data budget is determined by the number of steps taken and the number of tokens per step, for a total number of tokens seen over training. The model parameters is determined by counting the total number of trainable parameters in a model prior to training. The compute is approximated by how long the run took. All experiments are run on an 8 GPU DGX node consisting of 8 NVIDIA A5000 GPUs.

We train models for 1000 steps, enough for the models to start learning and to make their behavior/performance differentiable from other models. Every step, the model sees 1M tokens, Every experiment sees 1000 steps * 1M tokens per step = 1B tokens. The baseline model has around 36M parameters and by default every subsequent model matches this budget. 


# Results
More heads better (over number of layers, QKV dim, which layer), GLU better

## Our re-implementation compared to T5-base baseline
Our re-implementation has two differences compared to the T5-base baseline:
1. We use absolute position embeddings while T5 uses relative attention position embeddings.
    - Our position encodings require more parameters. For a max sequence length of 1024 and a dim_model of 768, we need 1024 * 768 ~ 800k parameters. For a relative attention num buckets of 32 and num_heads of 12, T5 uses 32 * 12 ~ 400 parameters.
2. Our tokenizer is trained only on wikitext-103 which transfers tokenization benefits to the training wikipedia dataset. This results in more efficient representations per token for our model and more productive training.

## More heads
Less layers - num layers from 6 to 4, num heads from 8 to 16. Slightly better, slightly faster, slightly more unstable (?)

Less QKV dim - qkv dim from 64 to 32, num heads from 8 to 16. 

Less QKV dim and less layers (4M less params) on par with baseline.

## LessHeadsMoreQKVDimExperiment
Dim qkv from 64 to 128, num heads from 8 to 4

Slightly worse, faster, maybe because there's like a minimum number of heads, then past that the model doesn't care.

## Scaling heads
More heads better, no matter which layer. Just keep QKV dim constant

## APESumOverAverageExperiment / APEUnitVarianceExperiment
Different recombinations of input and output for position encoding. Generally worse than the baseline

## GatedLinearUnit experiments
All slightly better than the baseline and slightly more expensive.

## Mixed Act
All slightly worse than the baseline, slightly more varied. 

## No Self-Attention residual
Makes training significantly worse and more varied, maybe we shouldn't sparsify attention with no residual

## Database experiments
Interesting idea, but not better in its current instantiation.

## Unified attention
Not very good, much less modeling capacity

## Big heads
Bigheads3 experiment -- perhaps too poorly conditioned, possible to revisit and improve.

## Baseline Large
The relative patterning of experiments stays the same when moving from the base exps 220M params to the large exps 740M params, very cool to see. 

# Ideas TODO
Sequence length reduction idea, every attention layer has (N, L, D) input but outputs (N, L, D_\prime). How would attention residuals work? No residuals on self attention with a deep network is disastrous. Maybe add a no-op in the keys and values? Have a zero value vector and a some corresponding key vector. The key vector could be learned or fixed. Previous work has tried zero key and zero value, but this is incorrect for bias-less models. Also a little bit hard to imagine for models with bias since in the softmax the dot product is 0. Would also be a crazy speedup

Position encoding -- implement T5 relative attention encoding and rotary embeddings. 

Run baseline on larger models, both T5 and custom implementation. 

Albert -- one big boy layer multiple times.

Encoder/decoder vs decoder-only paradigm. Would need a tad bit of dataset work for decoder-only, but it's just converting from text to tokens.

Bottlenecks -- force model summarization in not the L dim but the D dim. Could go from something like 768 to 384 to 192 layer by layer or even the inverse. 768 is a huge representation for a single token. Maybe have the scaled up dims the same i.e. attention is still the same 768 = 12 heads by 64 qkv dim but the intermediate dim is like 192 and feedforward is the same 192 to 768 * 4 and back down to 192. 

# Baseline V2

- [TODO] Relative position encoding - T5 and rotary
  - https://arxiv.org/abs/1803.02155
  - https://arxiv.org/abs/2009.13658
  - https://arxiv.org/abs/2104.09864
- GELU-GLU activation in FF
- Round both embedding and FF layers to multiple of 64
- No key/value cross attention map
