aux_save_dir: baseline3
storage_dir: C:\Users\GeorgeS\Documents\facultate\master\projects\Backforward-Propagation/data
save_dir: MNIST/MLP/baseline3_\2525
load_model: ''
load_optimizer: ''
load_training_state: ''
infer_only: false
half: false
grad_scaler: true
device: cuda:0
seed: 2525
progress_bar: true
save_model: true
save_interval: 10
epochs: 1000
val_every: 5
es_patience: 200
es_min_delta: 0.0001
es_metric: Train/crossentropyloss
scheduler_metric: Train/crossentropyloss
optimized_metric: Val/Accuracy
train_dataset:
  name: MNIST
  load_params:
    root: C:\Users\GeorgeS\Documents\facultate\master\projects\Backforward-Propagation/data
    train: true
    download: true
  save_in_memory: true
  shuffle: true
  batch_size: 128
  num_workers: 0
  pin_memory: true
  drop_last: true
  subset: 0
  update_every: 1
  transform: general/mnist
val_dataset:
  name: MNIST
  load_params:
    root: C:\Users\GeorgeS\Documents\facultate\master\projects\Backforward-Propagation/data
    train: false
    download: true
  save_in_memory: true
  shuffle: false
  batch_size: 512
  num_workers: 0
  pin_memory: true
  drop_last: false
  subset: 0.0
  transform: general/mnist
initialization: 2
initialization_batch_norm: true
loss:
  crossentropyloss:
    class_weights: null
    reduction: mean
train_metrics:
  Accuracy:
    parameters: null
    aggregator: mean
    levels:
    - epoch
    higher_is_better: true
  crossentropyloss:
    parameters: null
    aggregator: mean
    levels:
    - batch
    - epoch
    higher_is_better: false
val_metrics:
  Accuracy:
    parameters: null
    aggregator: mean
    levels:
    - epoch
    higher_is_better: true
  crossentropyloss:
    parameters: null
    aggregator: mean
    levels:
    - batch
    - epoch
    higher_is_better: false
solver_metrics:
  Model Norm:
    parameters:
      norm_type: 2
    aggregator: null
    levels:
    - epoch
  Learning Rate:
    parameters: null
    aggregator: null
    levels:
    - epoch
model:
  name: MLP
  parameters:
    input_size: 784
    hidden_size: 256
    output_size: 10
optimizer:
  name: SGD
  parameters:
    lr: 0.005
    weight_decay: 0.0
    momentum: 0.0
    nesterov: false
  max_norm: 0.0
  grad_penalty: 0.0
  batch_replay: false
  use_lookahead: false
  lookahead_k: 5
  lookahead_alpha: 0.5
  use_SAM: false
  SAM_rho: 0.5
scheduler:
  name: StaticScheduler
  StaticScheduler: {}

## Val/Accuracy
 0.98000