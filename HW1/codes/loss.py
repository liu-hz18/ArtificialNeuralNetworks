from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        return 0.5 * (np.square(target - input)).mean(axis=0).sum()
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        return (input - target) / len(input)
        # TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name
        self._saved_tensor = 1e-20

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        input -= np.max(input)
        exp_input = np.exp(input)
        self._saved_tensor = exp_input / (np.sum(exp_input, axis=1, keepdims=True) + 1e-20)
        return np.mean(np.sum(-target * np.log(self._saved_tensor + 1e-20), axis=1))
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        return (self._saved_tensor - target) / len(input)
        # TODO END


class HingeLoss(object):
    def __init__(self, name, threshold=0.05):
        self.name = name
        self.threshold = threshold
        self._saved_target_mask = None
        self._saved_input_update_mask = None

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        N, K = input.shape
        self._saved_target_mask = target > 0
        target_prob = np.resize(input[self._saved_target_mask].repeat(input.shape[1], axis=0), (N, K))
        temp = self.threshold - target_prob + input
        self._saved_input_update_mask = (temp > 0) & (~self._saved_target_mask)
        return np.sum(temp[self._saved_input_update_mask]) / N
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        back = np.zeros_like(input)
        delta = 1. / len(input)
        back[self._saved_target_mask] -= np.sum(self._saved_input_update_mask, axis=1) * delta
        back[self._saved_input_update_mask] += delta
        return back
        # TODO END
