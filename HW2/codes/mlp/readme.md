# Running Guide

#### Example:

```
python main.py −−num_epochs 50 −−drop_rate 0.5 −−data_dir ../cifar−10_data
```

#### Usage:

```
usage: main.py [-h] [--batch_size BATCH_SIZE] [--num_epochs NUM_EPOCHS]
               [--learning_rate LEARNING_RATE] [--drop_rate DROP_RATE]
               [--is_train IS_TRAIN] [--data_dir DATA_DIR]
               [--train_dir TRAIN_DIR] [--inference_version INFERENCE_VERSION]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Batch size for mini-batch training and evaluating.
                        Default: 100
  --num_epochs NUM_EPOCHS
                        Number of training epoch. Default: 20
  --learning_rate LEARNING_RATE
                        Learning rate during optimization. Default: 1e-3
  --drop_rate DROP_RATE
                        Drop rate of the Dropout Layer. Default: 0.5
  --is_train IS_TRAIN   True to train and False to inference. Default: True
  --data_dir DATA_DIR   Data directory. Default: ../cifar-10_data
  --train_dir TRAIN_DIR
                        Training directory for saving model. Default: ./train
  --inference_version INFERENCE_VERSION
                        The version for inference. Set 0 to use latest
                        checkpoint. Default: 0
```

