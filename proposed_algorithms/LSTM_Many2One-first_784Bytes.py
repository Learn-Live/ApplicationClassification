# -*- coding: utf-8 -*-
r"""
    Purpose : Application Classification by Sequence Models and Long-Short Term Memory Networks on the
              first n (1,2,3, ... ) packets.

    created at 20180715
"""

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable

from utilities.preprocess import idx_reader
from utilities.preprocess import TrafficDataset, split_train_test


def one_hot_sklearn(label_integer):
    label_integer = np.asarray(label_integer, dtype=int)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = label_integer.reshape(len(label_integer), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return np.array(onehot_encoded, dtype=int)


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = 1
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.num_layers)

        # proposed_algorithms = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
        # self.loss_function = nn.NLLLoss()
        self.loss_function = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(self.lstm.parameters(), lr=0.1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-4)
        #

        # The linear layer that maps from hidden state space to tag space
        self.tagset_size = tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden(num_layers=self.num_layers, batch_size=batch_size)

    def init_hidden(self, num_layers, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # return (torch.zeros(1, 1, self.hidden_dim),
        #         torch.zeros(1, 1, self.hidden_dim))

        return (
            torch.zeros(num_layers, batch_size, self.hidden_dim),
            # the last layer output, which is equal to out[-1]
            torch.zeros(num_layers, batch_size, self.hidden_dim))  # the last cell hidden state.

    def achieve_sentence(self, sentences, first_n_pkts=2):
        """
            input size of lstm
        :param sentences: [batch_size, len(sentence_i), input_size]
        :return:
        """
        # new_sentences=torch.Tensor()
        new_sentences = []
        for sentence_i in sentences:
            t = 0
            cnt = 1
            tmp_lst = []
            while (cnt - 1) * num_features + (cnt - 1) < len(sentence_i):
                tmp_lst.append(
                    sentence_i[(cnt - 1) * num_features + (cnt - 1): cnt * num_features + (cnt - 1)].data.tolist()[
                    :num_features])
                # t = (cnt - 1) * 60 + (cnt - 1)
                if cnt == first_n_pkts:
                    break
                cnt += 1
            tmp_lst = torch.from_numpy(np.array(tmp_lst))
            new_sentences.append(tmp_lst)

        new_sentences = torch.stack(new_sentences)

        return new_sentences

    def forward(self, sentences):
        # # Also, we need to clear out the hidden state of the LSTM,
        # # detaching it from its history on the last instance.
        self.hidden = self.init_hidden(num_layers=self.num_layers, batch_size=sentences.shape[0])

        # embeds = self.word_embeddings(sentence)
        # embeds = self.achieve_sentence(sentences).float()   # change double to float
        embeds = sentences
        # print('embed:',embeds)
        # # embeds.view(len(sentence), batch_size, input_size)
        # # Input needs to be a 3d tensor with dimensions (seq_len, batch_size, input_size), so: length of your
        # # input sequence, batch size, and number of features (if you only have the time series there is only 1 feature).
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(embeds[0]), sentences.shape[0], embeds.shape[-1]), self.hidden)
        # tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        # return tag_scores[-1]   # when tag_scores.shape > 1, only return the last cell output.
        tag_space = self.hidden2tag(
            lstm_out.view(len(embeds[0]), sentences.shape[0], -1)[-1])  # only return the last cell output.
        tag_scores = F.softmax(tag_space, dim=1)
        return tag_scores  # when tag_scores.shape > 1, only return the last cell output.
        # return tag_space

    ######################################################################
    # Train the proposed_algorithms:

    def train(self, train_loader):

        self.loss_hist = []
        # # See what the scores are before training
        # # Note that element i,j of the output is the score for tag j for word i.
        # # Here we don't need to train, so the code is wrapped in torch.no_grad()
        # with torch.no_grad():
        #     inputs = prepare_sequence(training_data[0][0], word_to_ix)
        #     tag_scores = proposed_algorithms(inputs)
        #     print(tag_scores)
        for epoch in range(EPOCHES):  # again, normally you would NOT do 300 epochs, it is toy input_data
            # print('epoch:', epoch)
            for step, (b_x, b_y) in enumerate(train_loader):
                # training_data = zip(X_train, y_train)
                # for sentence, tags in training_data:
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.lstm.zero_grad()

                b_x = self.achieve_sentence(b_x, first_n_pkts)
                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                # sentence_in = prepare_sequence(sentence, word_to_ix)
                # b_x = b_x.view([b_x.shape[0], -1])  # (nSamples, nChannels, x_Height, x_Width)
                b_x = Variable(b_x, requires_grad=True).float()
                b_y = Variable(b_y).type(torch.FloatTensor)
                sentence_in = torch.Tensor(b_x)
                # print('sentence_in:', sentence_in)
                # targets = prepare_sequence(tags, tag_to_ix)
                targets = torch.Tensor(b_y).long()
                # print('targets:', targets)

                # Step 3. Run our forward pass.
                tag_scores = self.forward(sentence_in)
                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = self.loss_function(tag_scores, targets)
                self.loss_hist.append(loss.tolist())
                if epoch % 50 == 0 and step == 0:
                    if epoch == 0 and step == 0:
                        print(
                            '---\'epoch, loss, batch[batch_size, first_n_pkts, input_size], (softmax, preds, reals)---\'')
                    # print('epoch :', epoch, ', loss :', loss, ', targets : ', targets, ', tag_scores :', tag_scores)
                    _, preds = torch.max(tag_scores.data, dim=1)
                    print('epoch :', epoch, ', loss->', loss, ', ', b_x.shape, ', targets->',
                          list(zip(tag_scores.data.tolist(), preds, targets)))  # softmax, preds, reals

                loss.backward()
                self.optimizer.step()

    def test(self, test_loader):

        # See what the scores are after training
        with torch.no_grad():
            # inputs = prepare_sequence(training_data[0][0], word_to_ix)
            # cnt = 0
            total = 0
            correct = 0.0
            for step, (b_x, b_y) in enumerate(test_loader):
                b_x = self.achieve_sentence(b_x, first_n_pkts)
                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                # sentence_in = prepare_sequence(sentence, word_to_ix)
                # b_x = b_x.view([b_x.shape[0], -1])  # (nSamples, nChannels, x_Height, x_Width)
                b_x = Variable(b_x).float()
                b_y = Variable(b_y).type(torch.FloatTensor)
                sentence_in = torch.Tensor(b_x)
                # print('sentence_in:', sentence_in)
                # targets = prepare_sequence(tags, tag_to_ix)
                b_y = torch.Tensor(b_y).long()
                # print('targets:', targets)

                # Step 3. Run our forward pass.
                b_y_preds = self.forward(sentence_in)
                # print('b_y_preds', b_y_preds.input_data.tolist())
                _, predicted = torch.max(b_y_preds.data, dim=1)
                total += b_y.size(0)
                correct += (predicted == b_y).sum().item()

                # # Step 4. Compute the loss, gradients, and update the parameters by
                # #  calling optimizer.step()
                # loss = self.loss_function(tag_scores, targets)

                if step == 0:
                    cm = confusion_matrix(b_y, predicted, labels=[i for i in range(num_classes)])
                    sk_accuracy = accuracy_score(b_y, predicted) * len(b_y)
                else:
                    cm += confusion_matrix(b_y, predicted, labels=[i for i in range(num_classes)])
                    sk_accuracy += accuracy_score(b_y, predicted) * len(b_y)

            print(cm, sk_accuracy / total)

        print('accuracy = ', correct / total)

        # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
        # for word i. The predicted tag is the maximum scoring tag.
        # Here, we can see the predicted sequence below is 0 1 2 0 1
        # since 0 is index of the maximum value of row 1,
        # 1 is the index of maximum value of row 2, etc.
        # Which is DET NOUN VERB DET NOUN, the correct sequence!
        # print(tag_scores)


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
        model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, '', num_classes)
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

    # # get input_data
    # X_train, y_train, X_test, y_test = read_skype_sample()
    #
    # # # stats
    # # print('Y :', Counter(np.concatenate([y_train, y_test])))
    # # print(
    # #     'X_train : %d, y_train : %d, label : %s' % (
    # #     X_train.shape[0], y_train.shape[0], dict(sorted(Counter(y_train).items()))))
    # # # print('y_train : %s\ny_test  : %s'%(Counter(y_train), Counter(y_test)))
    # # print('X_test  : %d, y_test  : %d, label : %s' % (
    # # X_test.shape[0], y_test.shape[0], dict(sorted(Counter(y_test).items()))))


if __name__ == '__main__':
    torch.manual_seed(1)

    n = 784
    name_str = 'facebook'
    # name_str = 'hangout'
    # name_str = 'skype'
    train_output_file, test_output_file = read_skype_sample(name_str, n)
    input_file = train_output_file

    remove_labels_lst = [1]
    input_file, num_c = remove_special_labels(input_file, remove_labels_lst)
    print(input_file)

    global batch_size, EPOCHES, num_classes, num_features
    batch_size = 200
    EPOCHES = 100
    num_classes = num_c
    num_features = 5
    run_main(input_file)
