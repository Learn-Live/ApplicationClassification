# -*- coding: utf-8 -*-
r"""
    Purpose : Application Classification by Sequence Models and Long-Short Term Memory Networks on the
              first n (10,20,30, ... ) bytes.

    created at 20180727
"""
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn

from utilities.preprocess import idx_reader
from utilities.preprocess import TrafficDataset, split_train_test


class LSTMBytes(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMBytes, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.num_hidden_layers = 2
        self.lstm_bytes = nn.LSTM(self.input_dim, self.hidden_dim,
                                  num_layers=self.num_hidden_layers)  # return output, hidden
        self.hidden2output = nn.Linear(self.hidden_dim, self.output_dim)  # fc layer

        self.loss_function = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.lstm_bytes.parameters(), lr=1e-3)

    def init_hidden(self, num_layers, batch_size):
        # Before we've done anything, we don't have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (
            torch.zeros(num_layers, batch_size, self.hidden_dim),  # output ->h(t)
            torch.zeros(num_layers, batch_size, self.hidden_dim))  # hidden state ->c(t).

    def forward(self, sequences):
        tmp_batch_size = sequences.shape[0]
        # init the first input hidden states
        hidden_state = self.init_hidden(num_layers=self.num_hidden_layers, batch_size=tmp_batch_size)

        tmp_sequences = self.achieve_sentence(sequences, first_n_pkts).float()
        # Input needs to be a 3d tensor with dimensions (seq_len, batch_size, input_size)
        lstm_out, hidden_state = self.lstm_bytes(
            tmp_sequences.view(tmp_sequences.shape[1], tmp_batch_size, self.input_dim), hidden_state)
        fc_output = self.hidden2output(lstm_out[-1])  # only return the last cell output.
        # output = F.softmax(fc_output, dim=1)
        return fc_output

    def achieve_sentence(self, sentences, first_n_pkts=2):
        """
            input size of lstm
        :param sentences: [batch_size, len(sentence_i), input_size]
        :return:
        """
        # new_sentences=torch.Tensor()
        new_sentences = []
        for sentence_i in sentences:
            cnt = 1
            tmp_lst = []
            while (cnt - 1) * num_features + (cnt - 1) < len(sentence_i):
                tmp_lst.append(
                    sentence_i[(cnt - 1) * num_features + (cnt - 1): cnt * num_features + (cnt - 1)].data.tolist()[
                    :num_features])
                if cnt == first_n_pkts:
                    break
                cnt += 1
            tmp_lst = torch.from_numpy(np.array(tmp_lst))
            new_sentences.append(tmp_lst)
        new_sentences = torch.stack(new_sentences)

        return new_sentences

    def train(self, data_loader):
        """

        :param data_loader:
        :return:
        """
        for epoch in range(epoches):
            for step, (b_x, b_y) in enumerate(data_loader):
                b_y_preds = self.forward(b_x)
                loss = self.loss_function(b_y_preds, b_y.long())

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if epoch % 50 == 0 and step == 0:
                    if epoch == 0 and step == 0:
                        print(
                            '---\'epoch, loss, batch[batch_size, first_n_pkts, input_size], (softmax, preds, reals)---\'')
                    # print('epoch :', epoch, ', loss :', loss, ', targets : ', targets, ', tag_scores :', tag_scores)
                    _, preds = torch.max(b_y_preds.data, dim=1)
                    print('epoch :', epoch, ', loss->', loss, ', ', b_x.shape, ', targets->',
                          list(zip(b_y_preds.data.tolist(), preds, b_y)))  # softmax, preds, reals

    def test(self, data_loader):
        """

        :param data_loader:
        :return:
        """
        with torch.no_grad():
            total = 0.0
            correct = 0
            for step, (b_x, b_y) in enumerate(data_loader):
                b_y_preds = self.forward(b_x)
                _, predicted = torch.max(b_y_preds.data, dim=1)
                total += b_y.size(0)
                correct += (predicted == b_y.long()).sum().item()
                if step == 0:
                    cm = confusion_matrix(b_y, predicted, labels=[i for i in range(num_classes)])
                    sk_accuracy = accuracy_score(b_y, predicted) * len(b_y)
                else:
                    cm += confusion_matrix(b_y, predicted, labels=[i for i in range(num_classes)])
                    sk_accuracy += accuracy_score(b_y, predicted) * len(b_y)
            print(cm, sk_accuracy / total)

        print('accuracy = ', correct / total)

        return correct / total


def show_figure(data):
    import matplotlib.pyplot as plt

    plt.plot(range(len(data)), data)
    plt.show()


def split_ptks(feature_file, label_file, ):
    """

    :param features_file:
    :param label_file:
    :return:
    """

    features = []
    labels = []
    with open(feature_file, 'r') as fid_in:
        line = fid_in.readline()
        while line:
            features.append(line.strip().split(','))
            line = fid_in.readline()

    with open(label_file, 'r') as fid_in:
        line = fid_in.readline()
        while line:
            labels.append(line.strip().split(','))
            line = fid_in.readline()

    return features, labels


def get_loader_iterators_contents(train_loader):
    X = []
    y = []
    for step, (b_x, b_y) in enumerate(train_loader):
        X.extend(b_x.data.tolist())
        y.extend(b_y.data.tolist())

    return X, y


def run_main(input_file):
    dataset = TrafficDataset(input_file, transform=None, normalization_flg=True)

    train_sampler, test_sampler = split_train_test(dataset, split_percent=0.9, shuffle=True)
    cntr = Counter(dataset.y)
    print('dataset: ', len(dataset), ' y:', sorted(cntr.items()))
    # train_loader = torch.utils.input_data.DataLoader(dataset, batch_size, shuffle=True, num_workers=4)  # use all dataset
    train_loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=4, sampler=train_sampler)
    X, y = get_loader_iterators_contents(train_loader)
    cntr = Counter(y)
    print('train_loader: ', len(train_loader.sampler), ' y:', sorted(cntr.items()))
    global test_loader
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4,
                                              sampler=test_sampler)
    X, y = get_loader_iterators_contents(test_loader)
    cntr = Counter(y)
    print('test_loader: ', len(test_loader.sampler), ' y:', sorted(cntr.items()))

    EMBEDDING_DIM = num_features  # input_size
    HIDDEN_DIM = 30
    # proposed_algorithms = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, '', num_classes)

    for i in range(1, 11):
        print('first_%d_pkts' % i)
        global first_n_pkts
        first_n_pkts = i
        model = LSTMBytes(EMBEDDING_DIM, HIDDEN_DIM, num_classes)
        model.train(train_loader)

        print('***train accuracy: ')
        model.test(train_loader)

        print('***test accuracy: ')
        model.test(test_loader)


def remove_special_labels(input_file, remove_labels_lst=[2, 3]):
    output_file = input_file + '_remove_labels.csv'
    y = []
    data = []
    with open(output_file, 'w') as fid_out:
        with open(input_file, 'r') as fid_in:
            line = fid_in.readline()
            while line:
                line_arr = line.strip().split(',')
                tmp_label = int(float(line_arr[-1]))
                if tmp_label in remove_labels_lst:
                    line = fid_in.readline()
                    continue
                else:
                    y.append(tmp_label)
                    # fid_out.write(line)
                    data.append(line)
                    line = fid_in.readline()

        # change labels to continues ordered values, such as 0,1,2,.. not 0,2,...
        new_labels = sorted(Counter(y).keys())
        # new_labels = [i for i in range(len(new_labels))]
        for data_i in data:
            line = data_i.strip().split(',')
            tmp_label = int(float(line[-1]))
            if tmp_label in new_labels:
                line = ','.join(line[:-1]) + ',' + str(new_labels.index(tmp_label)) + '\n'
                fid_out.write(line)
            else:
                print('tmp_label:', tmp_label)

    return output_file, len(new_labels)


def read_skype_sample(name_str='facebook', n=784):
    data_path = '../input_data/fixed-length-transport-layer-payload/session/{}'.format(name_str)
    train_images_file = '{}/{}-byte-payload-per-flow-{}-train-images-idx2-ubyte.gz'.format(data_path, n, name_str)
    train_labels_file = '{}/{}-byte-payload-per-flow-{}-train-labels-idx1-ubyte.gz'.format(data_path, n, name_str)
    test_images_file = '{}/{}-byte-payload-per-flow-{}-test-images-idx2-ubyte.gz'.format(data_path, n, name_str)
    test_labels_file = '{}/{}-byte-payload-per-flow-{}-test-labels-idx1-ubyte.gz'.format(data_path, n, name_str)
    # X_train, X_test = np.expand_dims(idx_reader.read_images(train_images_file), 1), np.expand_dims(
    #     idx_reader.read_images(test_images_file), 1)
    X_train, X_test = idx_reader.read_images(train_images_file), idx_reader.read_images(test_images_file)
    y_train, y_test = idx_reader.read_labels(train_labels_file), idx_reader.read_labels(test_labels_file)

    # return X_train, y_train, X_test, y_test
    train_output_file = '%s_%dBytes_train.csv' % (name_str, n)
    with open(train_output_file, 'w') as fid_out:
        (m, n) = X_train.shape
        for row in range(m):
            line = ''
            for col in range(n):
                line += str(X_train[row][col]) + ','
            line += str(int(y_train[row])) + '\n'
            fid_out.write(line)

    test_output_file = '%s_%dBytes_test.csv' % (name_str, n)
    with open(test_output_file, 'w') as fid_out:
        (m, n) = X_test.shape
        for row in range(m):
            line = ''
            for col in range(n):
                line += str(X_test[row][col]) + ','
            line += str(int(y_test[row])) + '\n'
            fid_out.write(line)

    return train_output_file, test_output_file


if __name__ == '__main__':
    torch.manual_seed(1)

    n = 1000
    name_str = 'facebook'
    # name_str = 'hangout'
    # name_str = 'skype'
    train_output_file, test_output_file = read_skype_sample(name_str, n)
    input_file = train_output_file

    remove_labels_lst = [1]
    input_file, num_c = remove_special_labels(input_file, remove_labels_lst)
    print(input_file)

    global batch_size, epoches, num_classes, num_features
    batch_size = 200
    epoches = 10000
    num_classes = num_c
    num_features = 5
    run_main(input_file)
