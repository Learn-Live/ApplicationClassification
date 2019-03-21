from collections import Counter

import numpy as np
import random


def random_selects_n_items_from_list(value_list, num=10):
    length = len(value_list)
    if num > length:
        return value_list

    idxs=random.choices(range(length), k=num)

    return [value_list[i] for i in idxs]


def load_npy_data(input_file, session_size=8000, balance_flg=True):
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
    input_file = '../input_data/trdata-8000B.npy'
    X, y = load_npy_data(input_file, session_size=2000)
    # X, y = load_new_npy(input_file)
    save_to_arff(X, y)
