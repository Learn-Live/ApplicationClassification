# -*- coding: utf-8 -*-
"""
    data preprocess

"""
from collections import Counter

__author__ = 'Learn-Live'

# standard library
import numpy as np

# third party library
from sklearn.model_selection import train_test_split


def load_data(input_file, separator=','):
    data = []
    label = []
    with open(input_file, 'r') as fid_in:
        line = fid_in.readline()
        while line:
            line_arr = line.split(separator)
            data.append(line_arr[:-1])
            label.append(line_arr[-1].split('\n')[0])
            line = fid_in.readline()

    return data, label


def compute_mean(data_lst):
    if len(data_lst) > 0:
        sum = 0.0
        for i in range(len(data_lst)):
            sum += float(data_lst[i])

        mean = sum / len(data_lst)
    else:
        mean = 1e-6

    return mean


def load_data_compute_mean(input_file, separator=','):
    data = []
    label = []
    with open(input_file, 'r') as fid_in:
        line = fid_in.readline()
        while line:
            line_arr = line.split(separator)
            first_n = int((len(line_arr) - 1) / 2)  # len(line_arr)-1  to exclude "class"
            # data.append(line_arr[:-1])
            pkts_mean = compute_mean(line_arr[:first_n])
            flow_dur = float(line_arr[first_n])
            intr_tm_mean = compute_mean(line_arr[first_n + 2:-1])   # line_arr[first_n+1] always is 0

            data.append([pkts_mean, flow_dur, intr_tm_mean])
            label.append(line_arr[-1].split('\n')[0])
            line = fid_in.readline()

    return data, label


def load_data_2(input_file, first_n_pkts=20, separator=','):
    first_n_pkts_lst = []
    flow_dur = []
    intr_tm_lst = []
    label = []
    with open(input_file, 'r') as fid_in:
        line = fid_in.readline()
        while line:
            line_arr = line.split(separator)

            first_n_pkts_lst.append(line_arr[0:first_n_pkts])
            flow_dur.append([line_arr[first_n_pkts]])
            intr_tm_lst.append(line_arr[first_n_pkts + 1:2 * first_n_pkts + 1])
            label.append(line_arr[-1].split('\n')[0])

            line = fid_in.readline()

    return [first_n_pkts_lst, flow_dur, intr_tm_lst], label


def normalize_data(X, range_value=[-1, 1], eps=1e-5):  # down=-1, up=1

    new_X = np.copy(X)

    mins = new_X.min(axis=0)  # column
    maxs = new_X.max(axis=0)

    rng = maxs - mins
    for i in range(rng.shape[0]):
        if rng[i] == 0.0:
            rng[i] += eps

    new_X = (new_X - mins) / rng * (range_value[1] - range_value[0]) + range_value[0]

    return new_X


def change_label(Y, label_dict={'BENIGN': 1, 'Others': 0}):
    label_cnt = Counter(Y)
    label_cnt = list(label_cnt.keys())
    new_Y = []

    for i in range(len(Y)):
        for j in range(len(label_cnt)):
            if Y[i] == label_cnt[j]:
                new_Y.append(j)
    return new_Y


def achieve_train_test_data(X, Y, train_size=0.7, shuffle=True):
    # X=np.asarray(X, dtype=float)
    # Y=np.asarray(Y)
    # X_train, X_test = train_test_split(Y, Y, train_size, shuffle)  # it's not right, why?
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=train_size, shuffle=shuffle)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    input_file = '../results/AUDIO_first_n_pkts_10_all_in_one_file.txt'
    X, Y = load_data(input_file, separator=',')
    X = normalize_data(np.asarray(X, dtype=float), range_value=[-1, 1], eps=1e-5)
    Y = change_label(Y)
    X_train, X_test, y_train, y_test = achieve_train_test_data(X, Y, train_size=0.2, shuffle=True)
