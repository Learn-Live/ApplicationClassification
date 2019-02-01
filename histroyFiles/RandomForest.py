# -*- coding: utf-8 -*-
"""

@ use ANN to classify

ref: https://raw.githubusercontent.com/MorvanZhou/PyTorch-Tutorial/master/tutorial-contents/401_CNN.py

Dependencies:
    torch: 0.4
    torchvision
    matplotlib

"""
from collections import Counter

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from utilities.preprocess import achieve_train_test_data, load_data, normalize_data, change_label

__author__ = 'Learn_live'

# library
# standard library

# third-party library
import torch

from sklearn.neural_network import MLPClassifier


class MLP():

    def __init__(self, *args, **kwargs):
        aplpha = 1
        self.epochs = kwargs['epochs']
        self.batch_size = kwargs['BATCH_SIZE']
        self.first_n_pkts = kwargs['first_n_pkts']
        self.out_size = kwargs['num_class']

        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 64, 32, 16, 8), random_state=1)
        # self.clf=RandomForestClassifier(max_depth=5, random_state=0)
        # self.clf=DecisionTreeClassifier(max_depth=5)
        # self.clf=SVC()

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
        #     lst_tmp.append(flow_dur[i].input_data.tolist())
        #     lst_tmp.extend(pkts_outputs[i].input_data.tolist())
        #     lst_tmp.extend(intr_outputs[i].input_data.tolist())
        #     new_X.append(lst_tmp)
        # # X = [pkts_outputs, flow_dur, intr_outputs]
        # new_X = torch.Tensor(new_X)
        # y_preds = self.classify_ann(new_X)
        # # _, y_preds=y_preds.input_data.max(dim=1) # get max value of each row
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


if __name__ == '__main__':
    torch.manual_seed(1)  # reproducible
    num = 42
    input_file = '../results/AUDIO_first_n_pkts_10_all_in_one_file.txt'
    input_file = '../results/AUDIO_first_n_pkts_20_all_in_one_file.txt'
    X, Y = load_data(input_file)
    X = normalize_data(np.asarray(X, dtype=float), range_value=[0, 1], eps=1e-5)
    Y = change_label(Y)
    X_train, X_test, y_train, y_test = achieve_train_test_data(X, Y, train_size=0.9, shuffle=True)

    ann = MLP(BATCH_SIZE=20, first_n_pkts=10, epochs=10, num_class=len(Counter(y_train)))
    # training_set = Data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))  # X, Y
    # one_hot_y_train=one_hot_sklearn(y_train)
    training_set = (X_train[:, 0:num], y_train)
    ann.train(training_set)

    # show_figure(ann.train_hist['loss'])

    Y_preds = ann.predict(X_train[:, 0:num])
    print(Counter(Y_preds))
    acc = ann.evaluate(y_train, Y_preds)
    print('training accuracy:', acc)

    Y_preds = ann.predict(X_test[:, 0:num])
    print(Counter(Y_preds))
    acc = ann.evaluate(y_test, Y_preds)
    print('testing accuracy:', acc)
