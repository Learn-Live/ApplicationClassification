from collections import Counter

import numpy as np
import random


def random_selects_n_items_from_list(value_list, num=10):
    length = len(value_list)
    if num > length:
        return value_list

    idxs=random.choices(range(length), k=num)

    return [value_list[i] for i in idxs]


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


def load_npy_data(input_file, session_size=8000, balance_flg=True, norm_flg=False):

    print(f'input_file:{input_file}')
    data = np.load(input_file)
    y, X = data[1:, 0], data[1:, 1]
    data_dict = {}
    for i, (x_tmp, y_tmp) in enumerate(zip(X, y)):
        if y_tmp not in data_dict.keys():
            data_dict[y_tmp] = []
        else:
            data_dict[y_tmp].append(x_tmp[0, :session_size].tolist())
    # data_stat=Counter(data_dict)
    X_new = []
    y_new = []

    for i, key in enumerate(data_dict):
        if balance_flg:
            # the number of samples for each application
            data_dict[key] = random_selects_n_items_from_list(data_dict[key], num=900)
            # data_dict[y_tmp] = random.choices(value, k=500)
        X_new.extend(data_dict[key])
        y_new.extend([key] * len(data_dict[key]))

    if norm_flg:
        range_value = [-0.1, 0.1]
        print(f'range_value:{range_value}')
        X_new= normalize_data(X_new, range_value, eps=1e-5)  # down=-1, up=1

    return np.asarray(X_new, dtype=float), np.asarray(y_new, dtype=int)


def save_to_arff(X, y):
    with open('data.arff', 'w') as out:
        for i, (x_tmp, y_tmp) in enumerate(zip(X, y)):
            if i == 0:
                out.write('@relation payload\n')
                for j, v_tmp in enumerate(x_tmp):
                    out.write('@attribute ' + '%s' % str(j) + ' numeric\n')
                out.write('@attribute class {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}\n')
                out.write('@data\n')
            line = ''
            for v_tmp in x_tmp:
                line += str(v_tmp) + ','
            line += str(y_tmp) + '\n'
            out.write(line)

def load_new_npy(input_file,session_size=8000, balance_flg=True):
    # new data
    truetrainX = np.load('../input_data/newX.npy')[1:]
    truetrainY = np.load('../input_data/newY.npy')[1:].reshape(-1)
    truetrainX = np.asarray(truetrainX)
    print(len(truetrainX))
    truetrainY = np.asarray(truetrainY)

    # data = np.load(input_file)
    # y, X = data[1:, 0], data[1:, 1]
    y, X = truetrainY, truetrainX
    data_dict = {}
    for i, (x_tmp, y_tmp) in enumerate(zip(X, y)):
        if y_tmp not in data_dict.keys():
            data_dict[y_tmp] = []
        else:
            data_dict[y_tmp].append(x_tmp[0, :session_size].tolist())
    # data_stat=Counter(data_dict)
    X_new = []
    y_new = []

    for i, key in enumerate(data_dict):
        if balance_flg:
            # the number of samples for each application
            data_dict[key] = random_selects_n_items_from_list(data_dict[key], num=1000)
            # data_dict[y_tmp] = random.choices(value, k=500)
        X_new.extend(data_dict[key])
        y_new.extend([key] * len(data_dict[key]))
    return np.asarray(X_new, dtype=float), np.asarray(y_new, dtype=int)

    return truetrainX, truetrainY

if __name__ == '__main__':
    input_file = '../input_data/trdata-8000B_payload.npy'
    input_file = '../input_data/trdata-8000B_header_payload_20190326.npy'
    X, y = load_npy_data(input_file, session_size=200)
    # X, y = load_new_npy(input_file)
    save_to_arff(X, y)
