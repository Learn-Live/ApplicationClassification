# -*- coding: utf-8 -*-
r"""
Sequence Models and Long-Short Term Memory Networks
===================================================

"""

# Author: Robert Guthrie
from collections import Counter
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder

from preprocess.data_preprocess import achieve_train_test_data, change_label, normalize_data
from preprocess import idx_reader

def one_hot_sklearn(label_integer):
    label_integer = np.asarray(label_integer, dtype=int)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = label_integer.reshape(len(label_integer), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return np.array(onehot_encoded, dtype=int)


#
# #
# # # Prepare data:
# #
# # def prepare_sequence(seq, to_ix):
# #     idxs = [to_ix[w] for w in seq]
# #     return torch.tensor(idxs, dtype=torch.long)
# #
# #
# # training_data = [
# #     ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
# #     ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
# # ]
# word_to_ix = {}
# # for sent, tags in training_data:
# #     for word in sent:
# #         if word not in word_to_ix:
# #             word_to_ix[word] = len(word_to_ix)
# print('word_to_ix:', word_to_ix)
# tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
#
# # These will usually be more like 32 or 64 dimensional.
# # We will keep them small, so we can see how the weights change as we train.


######################################################################
# Create the model:


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = 20
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.num_layers)

        # model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
        # self.loss_function = nn.NLLLoss()
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.SGD(self.lstm.parameters(), lr=0.1)
        #

        # The linear layer that maps from hidden state space to tag space
        self.tagset_size = tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden(num_layers=self.num_layers, batch_size=1)

    def init_hidden(self, num_layers, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # return (torch.zeros(1, 1, self.hidden_dim),
        #         torch.zeros(1, 1, self.hidden_dim))

        return (
            torch.zeros(1 * num_layers, batch_size, self.hidden_dim),
            # the last layer output, which is equal to out[-1]
            torch.zeros(1 * num_layers, batch_size, self.hidden_dim))  # the last cell hidden state.

    def forward(self, sentence):
        # # Also, we need to clear out the hidden state of the LSTM,
        # # detaching it from its history on the last instance.
        self.hidden = self.init_hidden(num_layers=self.num_layers, batch_size=1)

        # embeds = self.word_embeddings(sentence)
        embeds = sentence
        lstm_out, self.hidden = self.lstm(
            embeds.view(sentence.shape[0], 1, -1), self.hidden)
        # tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        # return tag_scores[-1]   # when tag_scores.shape > 1, only return the last cell output.
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1)[-1])  # only return the last cell output.
        tag_scores = F.softmax(tag_space, dim=0)
        return tag_scores  # when tag_scores.shape > 1, only return the last cell output.

    ######################################################################
    # Train the model:

    def train(self, X_train, y_train):

        self.loss_hist = []
        # # See what the scores are before training
        # # Note that element i,j of the output is the score for tag j for word i.
        # # Here we don't need to train, so the code is wrapped in torch.no_grad()
        # with torch.no_grad():
        #     inputs = prepare_sequence(training_data[0][0], word_to_ix)
        #     tag_scores = model(inputs)
        #     print(tag_scores)
        for epoch in range(50):  # again, normally you would NOT do 300 epochs, it is toy data
            # print('epoch:', epoch)
            t = 0
            training_data = zip(X_train, y_train)
            for sentence, tags in training_data:
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.lstm.zero_grad()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                # sentence_in = prepare_sequence(sentence, word_to_ix)
                sentence_in = torch.Tensor(sentence)
                # print('sentence_in:', sentence_in)
                # targets = prepare_sequence(tags, tag_to_ix)
                targets = torch.Tensor(tags)
                # print('targets:', targets)

                # Step 3. Run our forward pass.
                tag_scores = self.forward(sentence_in)
                if t % 100 == 0:
                    print('epoch :', epoch, ', tag_scores :', tag_scores)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = self.loss_function(tag_scores, targets)
                self.loss_hist.append(loss.tolist())
                loss.backward()
                self.optimizer.step()

                t += 1

    def predict(self, X_test, y_test):

        # See what the scores are after training
        with torch.no_grad():
            # inputs = prepare_sequence(training_data[0][0], word_to_ix)
            cnt = 0
            for i in range(len(X_test)):
                # print('X_test[%d]: len=%d'(i, len(X_test[i])))
                sentence = torch.Tensor(X_test[i])
                tag_scores = self.forward(sentence)
                # targets = torch.Tensor(y_test[i])
                # tag_pred=(tag_scores==(tag_scores.max())).nonzero().tolist()[0][0]
                # tag_pred = (tag_scores == (tag_scores.max())).tolist()
                tag_pred_value, tag_pred_idx = tag_scores.max(dim=0)
                if i % 100 == 0:
                    print('i =', i, tag_scores, y_test[i].tolist())
                    print('i =', i, tag_pred_idx, np.argmax(y_test[i].tolist()))
                if tag_pred_idx.data == torch.from_numpy(np.array(np.argmax(y_test[i].tolist()))):
                    cnt += 1

        print('accuracy = ', cnt / len(X_test))

        # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
        # for word i. The predicted tag is the maximum scoring tag.
        # Here, we can see the predicted sequence below is 0 1 2 0 1
        # since 0 is index of the maximum value of row 1,
        # 1 is the index of maximum value of row 2, etc.
        # Which is DET NOUN VERB DET NOUN, the correct sequence!
        # print(tag_scores)


def show_figure(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.plot(range(len(data)), data)
    plt.savefig('loss.pdf', format='pdf')

def read_skype_sample():
    data_path = sys.argv[1] if len(sys.argv) >= 2 else '../data'
    train_images_file = '{}/1pkts-subflow-skype-train-images-idx2-ubyte.gz'.format(data_path)
    train_labels_file = '{}/1pkts-subflow-skype-train-labels-idx1-ubyte.gz'.format(data_path)
    test_images_file = '{}/1pkts-subflow-skype-test-images-idx2-ubyte.gz'.format(data_path)
    test_labels_file = '{}/1pkts-subflow-skype-test-labels-idx1-ubyte.gz'.format(data_path)
    X_train, X_test = np.expand_dims(idx_reader.read_images(train_images_file), 1), np.expand_dims(idx_reader.read_images(test_images_file), 1)
    y_train, y_test = idx_reader.read_labels(train_labels_file), idx_reader.read_labels(test_labels_file)
    return X_train, y_train, X_test, y_test

def _main():

    # init parameters
    torch.manual_seed(1)
    n = 1

    # get data
    X_train, y_train, X_test, y_test = read_skype_sample()

    # stats
    print('Y :', Counter(np.concatenate([y_train, y_test])))
    print(
        'X_train : %d, y_train : %d, label : %s' % (X_train.shape[0], y_train.shape[0], dict(sorted(Counter(y_train).items()))))
    # print('y_train : %s\ny_test  : %s'%(Counter(y_train), Counter(y_test)))
    print('X_test  : %d, y_test  : %d, label : %s' % (X_test.shape[0], y_test.shape[0], dict(sorted(Counter(y_test).items()))))

    # set hyper parameters
    # shape of X_train, X_test: (num_samples, 1, features)
    EMBEDDING_DIM = X_train.shape[2]
    HIDDEN_DIM = 30

    # create model
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, 0, len(Counter(np.concatenate([y_train, y_test]))))

    # one-hot encode y_train
    y_train = one_hot_sklearn(y_train)

    # train
    model.train(X_train, y_train)

    # loss histogram
    show_figure(model.loss_hist)

    # train predict
    print('Training')
    model.predict(X_train, y_train)


    # one-hot encode y_test
    y_test = one_hot_sklearn(y_test)

    # test predict
    print('Test')
    model.predict(X_test, y_test)
    

if __name__ == '__main__':
    _main()