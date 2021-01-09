# Running Guide

### Example

```shell
python main.py --num_epochs 100 --batch_size 32 --layers 1 --units 300 --decode_strategy random --cell gru
```

### Usage

```shell
usage: main.py [-h] [--name NAME] [--num_epochs NUM_EPOCHS]
			   [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
			   [--test TEST] [--cell {rnn,gru,lstm}]
			   [--embed_units EMBED_UNITS] [--units UNITS] [--layers LAYERS]
		   	   [--data_dir DATA_DIR] [--wordvec_dir WORDVEC_DIR]
			   [--train_dir TRAIN_DIR] [--decode_strategy {random,top-p}]
			   [--temperature TEMPERATURE] [--max_probability MAX_PROBABILITY]
optional arguments:
	-h, --help            show this help message and exit
	--name NAME           Experiment name. Default: run
	--num_epochs NUM_EPOCHS
		Number of training epoch. Default: 20
	--batch_size BATCH_SIZE
		The number of batch_size. Default: 32
	--learning_rate LEARNING_RATE
		Learning rate during optimization. Default: 1e-3
	--test TEST           Evaluate the model with the specified name. Default:
		None
	--cell {rnn,gru,lstm}
		Type of RNN Cell. Default: gru
	--embed_units EMBED_UNITS
		Size of word embedding. Default: 300
	--units UNITS         Size of RNN. Default: 64
	--layers LAYERS       Number of layers of RNN. Default: 1
	--data_dir DATA_DIR   Data directory. Default: ../data
	--wordvec_dir WORDVEC_DIR
		Wordvector directory. Default: ../wordvec
	--train_dir TRAIN_DIR
		Training directory for saving model. Default: ./train
	--decode_strategy {random,top-p}
		The strategy for decoding. Can be "random" or "top-p".
		Default: random
	--temperature TEMPERATURE
		The temperature for decoding. Default: 1
	--max_probability MAX_PROBABILITY
		The p for top-p decoding. Default: 1
```

### Changes in Code

```
1. add '--cell' option in ArgumentParser, options are ['rnn', 'lstm', 'gru']
2. weight_decay = 0.0001
3. add dropout=0.2 after embedding layer, add dropout=0.5 bettween rnn layers(only when layers > 1).
```

