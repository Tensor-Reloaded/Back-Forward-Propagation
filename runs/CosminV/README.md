all_metrics: false
backforward_epochs: 200
backforward_lr: 0.05
backforward_train_batch_size: 128
cuda: true
cuda_device: 0
dataset: CIFAR-10
epoch: 1
es_patience: 10000
half: false
initialization: 0
initialization_batch_norm: false
load_model: ''
lr: 0.01
lr_gamma: 0.5
lr_milestones:
- 30
- 60
- 90
- 120
- 150
mixpo: 1
model: VGG11()
momentum: 0.9
nesterov: true
num_workers_test: 2
num_workers_train: 4
progress_bar: true
reduce_lr_delta: 0.02
reduce_lr_min_lr: 0.0005
reduce_lr_patience: 20
save_dir: CosminV
save_interval: 5
save_model: false
seed: null
test_batch_size: 128
train_batch_size: 128
train_subset: null
use_reduce_lr: false
wd: 0.0005

## Accuracy
 0.000%
## Accuracy
 0.000%
## Accuracy
 79.100%