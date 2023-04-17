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
