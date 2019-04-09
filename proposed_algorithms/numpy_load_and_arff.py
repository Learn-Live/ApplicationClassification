from collections import Counter

import numpy as np
import random

from collections import Counter

from imblearn.combine import SMOTEENN
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE  # doctest: +NORMALIZE_WHITESPACE


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
            print(f'rng[{i}]={rng[i]}')

    new_X = ((new_X - mins) / rng) * (range_value[1] - range_value[0]) + range_value[0]

    return new_X


def load_npy_data(input_file, session_size=8000, balance_flg=True, norm_flg=False, over_sample_flg=True):

    print(f'input_file:{input_file}')
    data = np.load(input_file)
    if over_sample_flg:
        if 'over_sample' not in input_file:
            input_file = over_sample_SMOTE(input_file)
            data = np.load(input_file)
        X_new = data[:, :session_size]
        y_new = data[:, -1]
    else:
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


def over_sample_SMOTE(input_file):
    data = np.load(input_file)
    y, X = data[1:, 0], data[1:, 1]

    data_dict = {}
    for i, (x_tmp, y_tmp) in enumerate(zip(X, y)):
        if y_tmp not in data_dict.keys():
            data_dict[y_tmp] = []
        else:
            data_dict[y_tmp].append(x_tmp[0, :].tolist())
    # data_stat=Counter(data_dict)
    X_new = []
    y_new = []
    balance_flg = True
    for i, key in enumerate(data_dict):
        if balance_flg:
            # the number of samples for each application
            data_dict[key] = random_selects_n_items_from_list(data_dict[key], num=3000)
            # data_dict[y_tmp] = random.choices(value, k=500)
        X_new.extend(data_dict[key])
        y_new.extend([key] * len(data_dict[key]))

    print('Original dataset shape %s' % Counter(y))
    print('new Original dataset shape %s' % Counter(y_new))
    sm = SMOTE(sampling_strategy='auto', random_state=42)
    X = np.asarray(X_new, dtype=np.uint8)
    y = np.asarray(y_new, dtype=int)
    X_new, y_new = sm.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y_new))

    value = []
    for x_tmp, y_tmp in zip(X_new, y_new):
        x_t = x_tmp.tolist()
        x_t.append(y_tmp.tolist())
        value.append(x_t)

    output = input_file + '_over_sample_data.npy'
    np.save(output, np.asarray(value, dtype=float))

    return output


def over_sample_demo(input_file, flg=True):
    from collections import Counter
    from sklearn.datasets import make_classification
    from imblearn.over_sampling import SMOTE  # doctest: +NORMALIZE_WHITESPACE

    if flg:
        X, y = make_classification(n_classes=2, class_sep=2,
                                   weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
                                   n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
        print('Original dataset shape %s' % Counter(y))
        sm = SMOTE(random_state=42)
        X_new, y_new = sm.fit_resample(X, y)
        print('Resampled dataset shape %s' % Counter(y_new))
    else:
        print(f'input_file:{input_file}')
        data = np.load(input_file)

        if 'over_sample' in input_file:
            X = data[:, :-1]
            y = data[:, -1]
        else:
            y, X = data[1:, 0], data[1:, 1]

        print('Original dataset shape %s' % Counter(y))
        sm = SMOTE(random_state=42)
        X = np.asarray(list(map(lambda x: x[0].tolist(), X)), dtype=np.uint8)
        y = np.asarray(list(map(lambda x: x, y)), dtype=int)
        X_new, y_new = sm.fit_resample(X, y)
        print('Resampled dataset shape %s' % Counter(y_new))

    value = []
    for x, y in zip(X_new, y_new):
        x_tmp = x.tolist()
        x_tmp.append(y.tolist())
        value.append(x_tmp)

    np.save('./over_sample_data.npy', np.asarray(value, dtype=float))



if __name__ == '__main__':
    # input_file = '../input_data/trdata-8000B_payload.npy'
    # input_file = '../input_data/trdata-8000B_header_payload_20190326.npy'
    # input_file = '/Users/kunyang/PycharmProjects/ApplicationClassification/input_data/trdata_P_8000.npy'
    # # input_file='./over_sample_data.npy'
    # input_file = over_sample_SMOTE(input_file)

    input_file = '/Users/kunyang/PycharmProjects/ApplicationClassification/input_data/trdata_P_8000.npy_over_sample_data.npy'
    X, y = load_npy_data(input_file, session_size=200)
    # # X, y = load_new_npy(input_file)
    save_to_arff(X, y)
