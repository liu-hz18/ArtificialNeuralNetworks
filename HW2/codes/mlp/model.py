# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features):
		super(BatchNorm1d, self).__init__()
		self.num_features = num_features

		# Parameters
		self.weight = nn.Parameter(torch.ones(num_features))
		self.bias = nn.Parameter(torch.zeros(num_features))
		# momentum for `Exponential moving average`, see "https://en.wikipedia.org/wiki/Moving_average"
		self.momentum = 0.9
		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))
		
		# Initialize your parameter

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			mean = torch.mean(input)
			var = torch.var(input, unbiased=False)
			input = (input - mean) / torch.sqrt(var + 1e-5)
			input = self.weight * input + self.bias
			self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean
			self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
		else:
			mean = self.running_mean
			var = self.running_var
			input = (input - mean) / torch.sqrt(var + 1e-5)
			input = self.weight * input + self.bias
		return input
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p
		self.remain_rate_reverse = 1. / (1. - self.p)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			prob = torch.empty(*(input.size()), device=self.device).fill_(self.p)
			mask = torch.bernoulli(prob)
			input[mask == 1.] = 0.
			input *= self.remain_rate_reverse
		return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		self.hidden_size = 784
		self.mlp = nn.Sequential(
			nn.Linear(32*32*3, self.hidden_size),
			BatchNorm1d(self.hidden_size),
			nn.ReLU(),
			Dropout(drop_rate),
			nn.Linear(self.hidden_size, 10),
		)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		y = y.long()
		logits = self.mlp(x)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
