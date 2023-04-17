# march
Model ARCHitecture experiments

Example setup on Colab:
```bash
%load_ext autoreload
%autoreload 2

!git clone https://github.com/bri25yu/march
%cd march
!git pull
!pip -q -q -q install -r requirements.txt
%cd ..

!huggingface-cli login

%reload_ext tensorboard
%tensorboard --logdir .

from march import run

run("BaselineWikitextExperiment")
```

# Story
All of the following experiments are over constant data budget, model parameters, and compute unless noted otherwise. The data budget is determined by the number of steps taken and the number of tokens per step, for a total number of tokens seen over training. The model parameters is determined by counting the total number of trainable parameters in a model prior to training. The compute is approximated by how long the run took. All experiments are run on an NVIDIA A100 40GB GPU on Google Colab. 

We train models for 1000 steps, enough for the models to start learning and to make their behavior/performance differentiable from other models. Every step, the model sees 1M tokens, Every experiment sees 1000 steps * 1M tokens per step = 1B tokens. The baseline model has around 36M parameters and by default every subsequent model matches this budget. 
