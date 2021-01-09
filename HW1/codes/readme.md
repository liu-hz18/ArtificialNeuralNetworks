# Usage

```
usage: run_mlp.py [-h] [--activ {relu,sigmoid,gelu,tanh}] [--lr LR]
				  [--loss {cross_entropy,enclidean,hinge}] 
				  [--nlayer {1,2}] [--wd WD] [--mo MO] [--bsz BSZ]
optional arguments:
	-h, --help            show this help message and exit
	--activ {relu,sigmoid,gelu,tanh}
				  	  activation function, default=relu
	--loss {cross_entropy,enclidean,hinge}
					  loss function, default=cross_entropy
	--nlayer {1,2}        layers of mlp, default=1
	--lr LR               learning rate, default=0.001
	--wd WD               weight decay, default=0.001
	--mo MO               momentum, default=0.9
	--bsz BSZ             batch size, default=200
```

For example:

```
python run_mlp.py --nlayer 1 --activ relu --loss cross_entropy --lr 0.01 --wd 0.001 --mo 0.9
```

you can also run it using **default** parameters:

```
python run_mlp.py
```

------



# Other Changes in Code

Add class `Sotfmax` and `Tanh` in `layers.py`

Add function `one_layer_mlp`, `two_layer_mlp` in `run_mlp.py`

Add `ArgumentParser` to change hyperparameters conveniently in`run_mlp.py`.

Add `save_dir` to save running datas in directory like "1layer_gelu_cross_entropy_0.01_0.001_0.9" in `run_mlp.py`, and also add `train_loss`, `train_acc`, `valid_loss`, `valid_acc`to record data and save `.npy` to file.

Change `train_net()`and `test_net()` to return accuracy and loss for convenience.

