# -*- coding: utf-8 -*-
r"""
    purpose: to identify pkts or flow progressively by sequence models (LSTM)

    Refer to : https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
"""

# standard libraries


from collections import Counter

import matplotlib.pyplot as plt
# third-party libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder

# local libraries
from preprocess.data_preprocess import achieve_train_test_data, change_label, normalize_data


# Prepare data
def load_sequence_data_by_tshark(first_n_pkts_input_file, separator=','):
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
            line_arr = line.split()
            if len(line_arr) < 9:
                print('skip: ', line[:-2])  # reomve '\n'
                line = fid_in.readline()
                continue
            data.append([line_arr[-3], line_arr[-2]])
            # print([line_arr[-3], line_arr[-3]])
            label.append(line_arr[-1].split('\n')[0])
            line = fid_in.readline()

    Y = change_label(label)
    new_data = normalize_data(np.asarray(data, dtype=float), range_value=[0, 1], eps=1e-5)
    X = []
    for idx in range(len(new_data)):
        line_arr = new_data[idx]
        # len_tmp = int(line_arr[4])  # length of pkts_list
        line_tmp = []
        # for i in range(1, len_tmp + 1):  # len(pkts_list), [1, len_tmp+1)
        #     if i == 1:
        #         line_tmp.append([line_arr[0],line_arr[1],line_arr[2], line_arr[3], line_arr[4 + i], line_arr[4 + len_tmp + i]])  # srcport, dstport, [pkts_lst[0], flow_duration]
        #     else:
        #         line_tmp.append([line_arr[0],line_arr[1],line_arr[2], line_arr[3],line_arr[4 + i], line_arr[
        #             4 + len_tmp + (i + 1)]])  # [pkts_lst[0], intr_tm_lst[1]], intr_tm_lst from 1, 2, ...

        # line_tmp=[line_arr[-3], line_arr[-3]]  # pkts_len
        line_tmp = [line_arr]
        X.append(line_tmp)

    return X, Y


def one_hot_sklearn(label_integer):
    """

    :param label_integer: E.g. [0, 1, 2, 3, ...]
    :return: one hot value
    """

    label_integer = np.asarray(label_integer, dtype=int)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = label_integer.reshape(len(label_integer), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return np.array(onehot_encoded, dtype=int)


# Create the model:
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        """

        :param embedding_dim: input dim of LSTM Cell
        :param hidden_dim: h_t or c_t
        :param vocab_size: not be used
        :param tagset_size: output size of Linear,  nn.Linear(hidden_dim, tagset_size)
        """
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
        """

        :param num_layers: number of cell in LSTM
        :param batch_size: not be used yet.
        :return: h_t_1, c_t_1
        """
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)

        return (
            torch.zeros(1 * num_layers, batch_size, self.hidden_dim),
            # h_t_1: the last layer output, which is equal to out[-1]
            torch.zeros(1 * num_layers, batch_size, self.hidden_dim))  # c_t_1: the last cell hidden state.

    def forward(self, sentence):
        """

        :param sentence: input value for Cell
        :return: output of LSTM
        """
        # # Also, we need to clear out the hidden state of the LSTM,
        # # detaching it from its history on the last instance.
        self.hidden = self.init_hidden(num_layers=self.num_layers, batch_size=1)

        # embeds = self.word_embeddings(sentence)
        embeds = sentence
        # print('embed:',embeds)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        # tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        # return tag_scores[-1]   # when tag_scores.shape > 1, only return the last cell output.
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1)[-1])  # only return the last cell output.
        tag_scores = F.softmax(tag_space, dim=0)
        return tag_scores  # when tag_scores.shape > 1, only return the last cell output.

    # Train the model:
    def train(self, X_train, y_train):
        """

        :param X_train: Features
        :param y_train: Label
        :return:
        """
        self.loss_hist = []  # used for log
        # # See what the scores are before training
        # # Note that element i,j of the output is the score for tag j for word i.
        # # Here we don't need to train, so the code is wrapped in torch.no_grad()
        # with torch.no_grad():
        #     inputs = prepare_sequence(training_data[0][0], word_to_ix)
        #     tag_scores = model(inputs)
        #     print(tag_scores)
        for epoch in range(2):  # again, normally you would NOT do 300 epochs, it is toy data
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
        """

        :param X_test:
        :param y_test:
        :return:
        """

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
                    print('i =', i, tag_pred_idx, y_test[i].tolist())
                if tag_pred_idx.data == y_test[i].tolist():
                    cnt += 1

        print('accuracy = ', cnt / len(X_test))


def show_figure(data):
    """

    :param data: []
    :return:
    """

    plt.plot(range(len(data)), data)
    plt.show()


def main_train_RNN(input_file, first_n=1):
    """

    :param input_file:
    :param first_n: compute from the first 1 pkts  to the first n pkts
    :return: the trained model
    """

    for i in range(1, first_n + 1):
        n = i
        # input_file = '../results/FILE-TRANS_CHAT_faceb_MAIL_gate__VIDEO_Yout/first_%d_pkts/%d_all_in_one.txt' % (n, n)
        # input_file = '../results/MAIL_gate__MAIL_gate__MAIL_Gatew/first_%d_pkts/%d_all_in_one.txt' % (n, n)
        print('input_file:', input_file)
        X, Y = load_sequence_data_by_tshark(input_file, separator='\t')
        print('Y :', Counter(Y))
        X_train, X_test, y_train, y_test = achieve_train_test_data(X, Y, train_size=0.7, shuffle=True)
        print(
            'X_train : %d, y_train : %d, label : %s' % (
                len(X_train), len(y_train), dict(sorted(Counter(y_train).items()))))
        # print('y_train : %s\ny_test  : %s'%(Counter(y_train), Counter(y_test)))
        print('X_test  : %d, y_test  : %d, label : %s' % (
            len(X_test), len(y_test), dict(sorted(Counter(y_test).items()))))

        EMBEDDING_DIM = len(X_train[0][0])  # input size of LSTM in cell
        HIDDEN_DIM = 20
        model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, 0, len(Counter(Y)))
        y_train = one_hot_sklearn(y_train)
        model.train(X_train[1:100], y_train[1:100])

        # show_figure(model.loss_hist)
        # model.predict(X_train, y_train)
        #
        # y_test = one_hot_sklearn(y_test)
        # model.predict(X_test, y_test)

    return model


def first_n_pkts_identify(X, model, threshold=0.8):
    """

    :param X: feature
    :param model: The trained LSTM Model
    :param threshold: to judge if to continue accumulate pkts again
    :return:
    """
    y_pred = model.forward(X)
    value, idx = y_pred.max(dim=0)
    flg = False
    if value.data.tolist() > threshold:  # value is Tensor.
        # print('--predict results : ', y_pred)
        flg = True

    return flg, y_pred


def online_identify(input_file='', model='', threshold=0.8):
    """

    :param input_file: mixed pkts pcaps
    :param model: the trained model
    :param threshold:
    :return:
    """
    flow_table = {}  # {'five_tuple':(flg, [pkt1, pkt2,..., pkt10 ])}
    identified_cnt = 0
    pkts_cnt = 0
    flow_cnt = 0
    with open(input_file, 'r') as fid_in:
        line_pkt = fid_in.readline()
        while line_pkt:
            pkts_cnt += 1
            line_arr = line_pkt.split()
            if len(line_arr) < 9:
                # print('skip: ',line_pkt)
                line_pkt = fid_in.readline()
                continue
            # No,  time,           srcIP     dstIP,    protocol, pkts_size, srcPort, dstPort
            #  1   0.000000    10.0.2.15 → 198.52.200.39 TLSv1.2 597 36922 443
            five_tuple = line_arr[2] + ',' + line_arr[4] + ',' + line_arr[-2] + ',' + line_arr[
                -1]  # do not includes 'protocol'
            pkt_data = [float(line_arr[-3]), float(line_arr[-3])]

            if five_tuple not in flow_table.keys():
                print('%s does not in flow_table' % five_tuple)
                flow_cnt += 1
                # flow_table[five_tuple]=(False,[pkt_data])
                flow_table[five_tuple] = (
                    False, -1, [pkt_data])  # identified or not, pred_value (softmax), pkts_data_lst
                line_pkt = fid_in.readline()
                continue

            (subflow_flg, pred_value, first_n_pkts_data) = flow_table[five_tuple]
            if not subflow_flg:
                first_n_pkts_data.append(pkt_data)
                # if first_n_pkts_identify(torch.Tensor(first_n_pkts_data),model):
                flg, pred_value = first_n_pkts_identify(torch.Tensor(first_n_pkts_data), model, threshold)
                if flg:
                    flow_table[five_tuple] = (
                        True, pred_value, first_n_pkts_data)  # identified or not, pred_value (softmax), pkts_data_lst
                    len_tmp = len(first_n_pkts_data)
                    identified_cnt += 1
                    print('*** %s is identified at the first %d pkts, the pred value is %s' % (
                        five_tuple, len_tmp, pred_value))

                else:
                    if len(first_n_pkts_data) > 10:  # and subflow_flg == False:
                        print(
                            '+After first %d pkts, %s [pkts_num = %d] still cannot be identified (pred_value=%s), so giving up, clear the pkts list' % (
                                10, five_tuple, len(flow_table[five_tuple][-1]), pred_value))
                        flow_table[five_tuple] = (subflow_flg, pred_value, [pkt_data])
                    else:
                        flow_table[five_tuple] = (subflow_flg, pred_value, first_n_pkts_data)
            else:
                # flow_table[five_tuple] = (True, first_n_pkts_data)
                identified_cnt += 1
                # print('%s is already identified!'%five_tuple)

            line_pkt = fid_in.readline()

        # print('Total num. of flow is %d, num. of identified flow is %d'%(len(flow_table.keys()),identified_cnt))
        print('(flow_cnt = %d) == (len(flow_table.keys()) = %d)' % (flow_cnt, len(flow_table.keys())))
        flow_identified_cnt = 0
        for key in flow_table.keys():
            flg, pred_value, _ = flow_table[key]
            # if flg == True and pred_value == true_label:
            if flg:
                print(flow_table[key])
                flow_identified_cnt += 1
        acc = flow_identified_cnt / len(flow_table.keys())
        print('Total num. of flow is %d, num. of identified pkts is %d' % (pkts_cnt, identified_cnt))
        print('Total num. of flow is %d, num. of identified pkts is %d, so accuracy is %f' % (
            len(flow_table.keys()), flow_identified_cnt, acc))


if __name__ == '__main__':
    torch.manual_seed(1)

    input_file_training = '../data/BROWSING_gate_SSL_Browsing_tshark.txt'
    model = main_train_RNN(input_file_training, first_n=2)

    input_file_Online_identify = '../data/AUDIO_spotifygateway_tshark.txt'
    online_identify(input_file_Online_identify, model, threshold=0.2)
