# -*- coding: utf-8 -*-
"""

@ use ANN to classify

ref: https://raw.githubusercontent.com/MorvanZhou/PyTorch-Tutorial/master/tutorial-contents/401_CNN.py

Dependencies:
    torch: 0.4
    torchvision
    matplotlib

"""
import os
from collections import Counter

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder

from torch import optim
from torch.autograd import Variable

from preprocess.data_preprocess import achieve_train_test_data, load_data, normalize_data, change_label

__author__ = 'Learn_live'

# library
# standard library
import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier


class MLP():

    def __init__(self, *args, **kwargs):
        aplpha = 1
        self.epochs = kwargs['epochs']
        self.batch_size = kwargs['BATCH_SIZE']
        self.first_n_pkts = kwargs['first_n_pkts']
        self.out_size = kwargs['num_class']

        first_n_pkts = 10
        self.small_in_size = first_n_pkts
        self.small_h_size = 5
        self.small_out_size = 2
        self.clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(128, 64, 32, 16, 8), random_state=1,
                                 activation='tanh')

    def train(self, training_set):
        X = training_set[0]
        y = training_set[1]
        # pkts_x = X[:, 0:self.first_n_pkts]
        # flow_dur = X[:, self.first_n_pkts]
        # intr_x = X[:, self.first_n_pkts + 1:2 * self.first_n_pkts + 1]
        #
        # pkts_outputs = self.pkts_ann(pkts_x)
        # # flow_dur = flow_dur
        # intr_outputs = self.intr_tm_ann(intr_x)
        #
        # new_X = []
        # for i in range(len(X)):
        #     lst_tmp = []
        #     lst_tmp.append(flow_dur[i].data.tolist())
        #     lst_tmp.extend(pkts_outputs[i].data.tolist())
        #     lst_tmp.extend(intr_outputs[i].data.tolist())
        #     new_X.append(lst_tmp)
        # # X = [pkts_outputs, flow_dur, intr_outputs]
        # new_X = torch.Tensor(new_X)
        # y_preds = self.classify_ann(new_X)
        # # _, y_preds=y_preds.data.max(dim=1) # get max value of each row
        #
        # return y_preds
        #
        self.clf.fit(X, y)

    def predict(self, X):
        # self.clf.predict([[2., 2.], [-1., -2.]])

        return self.clf.predict(X)

    def evaluate(self, Y, Y_preds):
        # cnt = 0
        # for i in range(len(Y)):
        #     if Y[i] == Y_preds[i]:
        #         cnt += 1
        # accuracy = cnt / len(Y)

        print(classification_report(Y, Y_preds))
        print(confusion_matrix(Y, Y_preds))

        return accuracy_score(Y, Y_preds)


def print_network(describe_str, net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(describe_str, net)
    print('Total number of parameters: %d' % num_params)


class ANN(nn.Module):

    def __init__(self, *args, **kwargs):
        # super(ANN,self).__init__() # python 2.x
        super().__init__()  # python 3.x

        self.epochs = kwargs['epochs']
        self.batch_size = kwargs['BATCH_SIZE']
        self.first_n_pkts = kwargs['first_n_pkts']
        self.out_size = kwargs['num_class']

        first_n_pkts = 10
        self.small_in_size = first_n_pkts
        self.small_h_size = 5
        self.small_out_size = 2

        self.pkts_ann = nn.Sequential(nn.Linear(self.small_in_size, self.small_h_size * 2), nn.Tanh(),
                                      nn.Linear(self.small_h_size * 2, self.small_h_size), nn.Tanh(),
                                      nn.Linear(self.small_h_size, self.small_out_size)
                                      )

        self.intr_tm_ann = nn.Sequential(nn.Linear(self.small_in_size, self.small_h_size * 2), nn.Tanh(),
                                         nn.Linear(self.small_h_size * 2, self.small_h_size), nn.Tanh(),
                                         nn.Linear(self.small_h_size, self.small_out_size)
                                         )

        self.in_size = 2 * self.small_out_size + 1  # first_n_pkts_list, flow_duration, intr_time_list
        self.h_size = 5
        # self.out_size = 1  # number of label, one-hot coding
        self.classify_ann = nn.Sequential(nn.Linear(self.in_size, self.h_size * 2), nn.Tanh(),
                                          nn.Linear(self.h_size * 2, self.h_size), nn.Tanh(),
                                          nn.Linear(self.h_size, self.out_size, nn.Softmax())
                                          )

        print('---------- Networks architecture -------------')
        print_network('pkts_ann:', self.pkts_ann)
        print_network('intr_tm_ann:', self.intr_tm_ann)
        print_network('classify_ann:', self.classify_ann)
        print('-----------------------------------------------')

        # self.criterion = nn.MSELoss(size_average=False)
        self.criterion = nn.MultiLabelMarginLoss()
        self.d_learning_rate = 1e-4
        self.g_learning_rate = 1e-4
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.Adam([self.pkts_ann, self.intr_tm_ann, self.classify_ann], lr=self.d_learning_rate,
        #                             betas=(0.5, 0.9))
        params = list(self.pkts_ann.parameters()) + list(self.intr_tm_ann.parameters()) + list(
            self.classify_ann.parameters())
        self.optimizer = optim.Adam(params, lr=self.g_learning_rate, betas=(0.5, 0.9))

    def forward(self, X):
        pass

    def train(self, training_set):
        self.train_hist = {}
        self.train_hist['loss'] = []

        # dataset = Data.TensorDataset(torch.Tensor(training_set[0]), torch.Tensor(training_set[1]))  # X, Y
        ### re divide dataset
        train_loader = Data.DataLoader(
            dataset=training_set,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,
            num_workers=2,
        )
        for epoch in range(self.epochs):
            for step, (b_x, b_y) in enumerate(
                    train_loader):  # type: (int, (object, object)) # gives batch data, normalize x when iterate train_loader
                # print('step:',step, ', batchs:',int(len(dataset)/self.batch_size))
                b_x = Variable(b_x, requires_grad=True)
                # b_y = Variable(b_y.view(-1, 1))
                b_y = Variable(b_y.long())
                y_preds = self.forward(b_x)
                loss = self.criterion(y_preds, b_y)  # net_outs, y_real(targets)

                self.optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                self.optimizer.step()  # apply gradients

                self.train_hist['loss'].append(loss.data.tolist())
                if step % 100 == 0:
                    print('epoch = %d, loss = %f' % (epoch, loss.data.tolist()))

    def predict(self, X):
        y_preds = self.forward(X)
        _, y_ = y_preds.data.max(dim=1, keepdim=False)  # return max_value as predicted value

        # y_preds=y_preds.data.tolist()
        # y_=[]
        # for i in range(len(y_preds)):
        #     if y_preds[i][0] > 0.5:
        #         y_.append(1)
        #     else:
        #         y_.append(0)

        return y_.data.tolist()

    def evaluate(self, Y, Y_preds):
        cnt = 0
        for i in range(len(Y)):
            if Y[i] == Y_preds[i]:
                cnt += 1
        accuracy = cnt / len(Y)

        return accuracy


def show_figure(loss):
    data = list(loss)
    plt.plot(range(len(data)), data)
    plt.show()


def one_hot_sklearn(label_integer):
    label_integer = np.asarray(label_integer, dtype=int)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = label_integer.reshape(len(label_integer), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return np.array(onehot_encoded, dtype=int)


if __name__ == '__main__':
    torch.manual_seed(1)  # reproducible

    input_file = '../results/AUDIO_first_n_pkts_10_all_in_one_file.txt'
    X, Y = load_data(input_file)
    X = normalize_data(np.asarray(X, dtype=float), range_value=[0, 1], eps=1e-5)
    Y = change_label(Y)
    X_train, X_test, y_train, y_test = achieve_train_test_data(X, Y, train_size=0.9, shuffle=True)

    ann = MLP(BATCH_SIZE=20, first_n_pkts=10, epochs=10, num_class=len(Counter(y_train)))
    # training_set = Data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))  # X, Y
    one_hot_y_train = one_hot_sklearn(y_train)
    training_set = (X_train, y_train)
    ann.train(training_set)

    # show_figure(ann.train_hist['loss'])

    Y_preds = ann.predict(X_train)
    print(Counter(Y_preds))
    acc = ann.evaluate(y_train, Y_preds)
    print('training accuracy:', acc)

    Y_preds = ann.predict(X_test)
    print(Counter(Y_preds))
    acc = ann.evaluate(y_test, Y_preds)
    print('testing accuracy:', acc)
