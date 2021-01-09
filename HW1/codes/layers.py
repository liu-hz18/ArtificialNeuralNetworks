import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor


class Tanh(Layer):
    def __init__(self, name):
        super(Tanh, self).__init__(name)

    def forward(self, input):
        exp_input = np.exp(input)
        exp_input_reciprocal = 1. / exp_input
        self._saved_for_backward((exp_input - exp_input_reciprocal) / (exp_input + exp_input_reciprocal))
        return self._saved_tensor

    def backward(self, grad_output):
        return grad_output * (1.0 - np.square(self._saved_tensor))


class Softmax(Layer):
    def __init__(self, name):
        super(Softmax, self).__init__(name)

    def forward(self, input):
        input -= np.max(input)
        exp_input = np.exp(input)
        self._saved_for_backward(exp_input / (np.sum(exp_input, axis=1, keepdims=True) + 1e-20))
        return self._saved_tensor

    def backward(self, grad_output):
        return self._saved_tensor * (1. - self._saved_tensor) * grad_output


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        self._saved_for_backward(input)
        return np.maximum(0, input)
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        return grad_output * (self._saved_tensor > 0)
        # TODO END


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        self._saved_for_backward(1. / (1. + np.exp(-input)))
        return self._saved_tensor
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        return grad_output * self._saved_tensor * (1. - self._saved_tensor)
        # TODO END


class Gelu(Layer):
    def __init__(self, name):
        super(Gelu, self).__init__(name)
        self._saved_input = None
        self.factor = np.sqrt(2 / np.pi)
        self.alpha = 0.044715
        self.backward_alpha = self.alpha * 3

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        self._saved_input = input
        self._saved_for_backward(np.tanh(self.factor * (input + self.alpha * np.power(input, 3))))
        return 0.5 * input * (1. + self._saved_tensor)
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        input = self._saved_input
        saved = self._saved_tensor
        return grad_output * 0.5 * ((1. + saved) + input * (
                self.factor * (1.0 - np.square(saved)) * (1.0 + self.backward_alpha * np.square(input))
            ))
        # TODO END


class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        self._saved_for_backward(input)
        return input.dot(self.W) + self.b
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        self.grad_W = self._saved_tensor.T.dot(grad_output)
        self.grad_b = grad_output.sum(axis=0)
        return grad_output.dot(self.W.T)
        # TODO END

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
