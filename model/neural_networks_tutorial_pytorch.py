# -*- coding: utf-8 -*-
"""
Neural Networks
===============

Neural networks can be constructed using the ``torch.nn`` package.

Now that you had a glimpse of ``autograd``, ``nn`` depends on
``autograd`` to define models and differentiate them.
An ``nn.Module`` contains layers, and a method ``forward(input)``\ that
returns the ``output``.

For example, look at this network that classifies digit images:

.. figure:: /_static/img/mnist.png
   :alt: convnet

   convnet

It is a simple feed-forward network. It takes the input, feeds it
through several layers one after the other, and then finally gives the
output.

A typical training procedure for a neural network is as follows:

- Define the neural network that has some learnable parameters (or
  weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule:
  ``weight = weight - learning_rate * gradient``

Define the network
------------------

Let’s define this network:
"""
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess.data_preprocess import achieve_train_test_data


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # self.conv1 = nn.Conv2d(1, 6, 5)
        # non-square kernels and unequal stride and with padding
        self.conv1 = nn.Conv2d(1, 6, (5, 1), stride=1)
        self.conv2 = nn.Conv2d(6, 16, (5, 1), stride=1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 1))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 1))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

########################################################################
# You just have to define the ``forward`` function, and the ``backward``
# function (where gradients are computed) is automatically defined for you
# using ``autograd``.
# You can use any of the Tensor operations in the ``forward`` function.
#
# The learnable parameters of a model are returned by ``net.parameters()``

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight


########################################################################
# Let try a random 32x32 input
# Note: Expected input size to this net(LeNet) is 32x32. To use this net on
# MNIST dataset, please resize the images from the dataset to 32x32.

# Prepare data
def load_sequence_data(first_n_pkts_input_file, separator=','):
    """

    :param first_n_pkts_input_file: E.g. input_file = '../results/AUDIO_first_n_pkts_10_all_in_one_file.txt'
    :param separator:
    :return: X=Features, Y=Label
    """

    data = []
    label = []
    with open(first_n_pkts_input_file, 'r') as fid_in:
        line = fid_in.readline()
        while line:
            # No,  time,           srcIP     dstIP,    protocol, pkts_size, srcPort, dstPort
            line_arr = line.split(',')
            if len(line_arr) < 9:
                print('skip: ', line[:-2])  # reomve '\n'
                line = fid_in.readline()
                continue
            data.append(list(np.asarray(line_arr, dtype=float)))
            # print([line_arr[-3], line_arr[-3]])
            label.append(line_arr[-1].split('\n')[0])
            line = fid_in.readline()

    # Y = change_label(label)
    Y = label
    # new_data = normalize_data(np.asarray(data, dtype=float), range_value=[0, 1], eps=1e-5)
    X = data
    # for idx in range(len(new_data)):
    #     line_arr = new_data[idx]
    #     # len_tmp = int(line_arr[4])  # length of pkts_list
    #     line_tmp = []
    #     # for i in range(1, len_tmp + 1):  # len(pkts_list), [1, len_tmp+1)
    #     #     if i == 1:
    #     #         line_tmp.append([line_arr[0],line_arr[1],line_arr[2], line_arr[3], line_arr[4 + i], line_arr[4 + len_tmp + i]])  # srcport, dstport, [pkts_lst[0], flow_duration]
    #     #     else:
    #     #         line_tmp.append([line_arr[0],line_arr[1],line_arr[2], line_arr[3],line_arr[4 + i], line_arr[
    #     #             4 + len_tmp + (i + 1)]])  # [pkts_lst[0], intr_tm_lst[1]], intr_tm_lst from 1, 2, ...
    #
    #     # line_tmp=[line_arr[-3], line_arr[-3]]  # pkts_len
    #     line_tmp = [line_arr]
    #     X.append(line_tmp)

    return X, Y


input_file = '../data/data_split_train_v2_711/train_1pkt_images_merged.csv'
print('input_file:', input_file)
X, Y = load_sequence_data(input_file)
print('Y :', Counter(Y))
X_train, X_test, y_train, y_test = achieve_train_test_data(X, Y, train_size=0.7, shuffle=True)
print(
    'X_train : %d, y_train : %d, label : %s' % (len(X_train), len(y_train), dict(sorted(Counter(y_train).items()))))
# print('y_train : %s\ny_test  : %s'%(Counter(y_train), Counter(y_test)))
print('X_test  : %d, y_test  : %d, label : %s' % (len(X_test), len(y_test), dict(sorted(Counter(y_test).items()))))
# dict(sorted(d.items()))

# input_sample = torch.randn(1, 1, 32, 1)
#
# target = torch.arange(1, 5)  # a dummy target, for example
# target = target.view(1, -1)  # make it the same shape as output

X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

for i in range(500):
    # in your training loop:
    optimizer.zero_grad()  # zero the gradient buffers
    output = net(X_train)
    loss = criterion(output, y_train)
    print('%d: loss=%s' % (i, loss))
    loss.backward()
    optimizer.step()  # Does the update

###############################################################
# .. Note::
#
#       Observe how gradient buffers had to be manually set to zero using
#       ``optimizer.zero_grad()``. This is because gradients are accumulated
#       as explained in `Backprop`_ section.
