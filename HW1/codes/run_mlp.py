from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear, Gelu, Softmax, Tanh
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss, HingeLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d

import os
import time
import numpy as np
from argparse import ArgumentParser

# Your model defintion here
# You should explore different model architecture


def one_layer_mlp(loss='enclidean', activation='relu'):
    model = Network()
    model.add(Linear('fc1', 784, 360, 0.01))
    model.add(activation_map[activation]('activation1'))
    model.add(Linear('fc2', 360, 10, 0.01))
    loss = loss_map[loss](name=loss)
    return model, loss


def two_layer_mlp(loss='enclidean', activation='relu'):
    model = Network()
    model.add(Linear('fc1', 784, 512, 0.01))
    model.add(activation_map[activation]('activation1'))
    model.add(Linear('fc2', 512, 256, 0.01))
    model.add(activation_map[activation]('activation2'))
    model.add(Linear('fc3', 256, 10, 0.01))
    loss = loss_map[loss](name=loss)
    return model, loss


parser = ArgumentParser()
parser.add_argument('--activ', type=str, default='relu', choices=['relu', 'sigmoid', 'gelu', 'tanh'], help='activation function, default=relu')
parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'enclidean', 'hinge'], help='loss function, default=cross_entropy')
parser.add_argument('--nlayer', type=int, default=1, choices=[1, 2], help='layers of mlp, default=1')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.01')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay, default=0.001')
parser.add_argument('--mo', type=float, default=0.9, help='momentum, default=0.9')
parser.add_argument('--bsz', type=int, default=200, help='batch size, default=200')
opt = parser.parse_args()
print(opt)

save_dir = str(opt.nlayer) + 'layer_' + opt.activ + '_' + opt.loss + '_' + str(opt.lr) + '_' + str(opt.wd) + '_' + str(opt.mo)
os.makedirs(save_dir, exist_ok=True)
save_dir += '/'

train_data, test_data, train_label, test_label = load_mnist_2d('data')

loss_map = {'enclidean': EuclideanLoss, 'cross_entropy': SoftmaxCrossEntropyLoss, 'hinge': HingeLoss}
activation_map = {'relu': Relu, 'sigmoid': Sigmoid, 'gelu': Gelu, 'tanh': Tanh}

if opt.nlayer == 1:
    model, loss = one_layer_mlp(loss=opt.loss, activation=opt.activ)
else:
    model, loss = two_layer_mlp(loss=opt.loss, activation=opt.activ)

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.
config = {
    'learning_rate': opt.lr,
    'weight_decay': opt.wd,
    'momentum': 0.9,
    'batch_size': opt.bsz,
    'max_epoch': 100,
    'disp_freq': 50,
    'test_epoch': 1
}


train_loss = []
train_acc = []
valid_loss = []
valid_acc = []

begin_time = time.time()
for epoch in range(config['max_epoch']):
    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        losses, acc = test_net(model, loss, test_data, test_label, config['batch_size'])
        valid_loss.append(losses)
        valid_acc.append(acc)

    LOG_INFO('Training @ %d epoch...' % (epoch))
    losses, acc = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
    train_loss.extend(losses)
    train_acc.extend(acc)

duration = time.time() - begin_time
with open(save_dir + 'info.log', 'w', encoding='utf-8') as f:
    f.write("Time elasped: " + str(duration) + " s\n")
    f.write('Best Train Loss: ' + str(min(train_loss)) + '\n')
    f.write('Best Train Acc: ' + str(max(train_acc)) + '\n')
    f.write('Best Test Loss: ' + str(min(valid_loss)) + '\n')
    f.write('Best Test Acc: ' + str(max(valid_acc)) + '\n')

train_loss = np.array(train_loss)
np.save(save_dir + 'train_loss.npy', train_loss)
train_acc = np.array(train_acc)
np.save(save_dir + 'train_acc.npy', train_acc)
valid_loss = np.array(valid_loss)
np.save(save_dir + 'valid_loss.npy', valid_loss)
valid_acc = np.array(valid_acc)
np.save(save_dir + 'valid_acc.npy', valid_acc)
