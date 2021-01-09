# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm2d(nn.Module):
	# TODO START
	def __init__(self, num_features):
		super(BatchNorm2d, self).__init__()
		self.num_features = num_features

		# Parameters
		self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
		self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
		# momentum for `Exponential moving average`, see "https://en.wikipedia.org/wiki/Moving_average"
		self.momentum = 0.9
		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))
		
		# Initialize your parameter

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		_, C, H, W = input.shape
		# remain dimension: C
		input_flat = input.permute(1, 0, 2, 3).reshape(C, -1)

		if self.training:
			mean = torch.mean(input_flat, dim=1)
			var = torch.var(input_flat, unbiased=False, dim=1)
			# input = (input - mean.view(1, C, 1, 1)) / torch.sqrt(var.view(1, C, 1, 1) + 1e-5)
			input = (input - mean.view(1, C, 1, 1)) / torch.sqrt(var.view(1, C, 1, 1) + 1e-5)
			input = (self.weight * input + self.bias)
			self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean
			self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
		else:
			mean = self.running_mean
			var = self.running_var
			input = (input - mean.view(1, C, 1, 1)) / torch.sqrt(var.view(1, C, 1, 1) + 1e-5)
			input = (self.weight * input + self.bias)
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
		# input: [batch_size, num_feature_map, height, width]
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
		self.out_channels = [300, 300]
		self.kernel_sizes = [5, 3]
		self.cnn_out_size = (((32 - self.kernel_sizes[0] + 1) // 2) - self.kernel_sizes[1] + 1) // 2
		# out = (in - fliter + 2*padding) // stride + 1
		# for default torch definition:
		#	Conv2d:     out = in - fliter + 1
		#   MaxPool2d:  out = in // fliter
		# in: 3 x 32 x 32
		self.cnn = nn.Sequential(
			nn.Conv2d(3, self.out_channels[0], self.kernel_sizes[0]),  # out: 300 x 28 x 28
			BatchNorm2d(self.out_channels[0]),
			nn.ReLU(),
			Dropout(drop_rate),
			nn.MaxPool2d(2),  # out: 300 x 14 x 14
			nn.Conv2d(self.out_channels[0], self.out_channels[1], self.kernel_sizes[1]),  # out: 300 x 12 x 12
			BatchNorm2d(self.out_channels[1]),
			nn.ReLU(),
			Dropout(drop_rate),
			nn.MaxPool2d(2),  # out: 300 x 6 x 6 = 10800
		)
		self.final_layer = nn.Linear(self.out_channels[1] * self.cnn_out_size * self.cnn_out_size, 10)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# TODO START
		# the 10-class prediction output is named as "logits"
		y = y.long()
		feature = self.cnn(x)
		logits = self.final_layer(feature.view(feature.shape[0], -1))
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
