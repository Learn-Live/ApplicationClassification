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
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from torch import optim
from torch.autograd import Variable

from utilities.preprocess import achieve_train_test_data, load_data, normalize_data, change_label

__author__ = 'Learn_live'

# library
# standard library

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt


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

        first_n_pkts = 1
        self.small_in_size = first_n_pkts
        self.small_h_size = 5
        self.small_out_size = 2

        # self.pkts_ann = nn.Sequential(nn.Linear(self.small_in_size, self.small_h_size * 2), nn.Tanh(),
        #                               nn.Linear(self.small_h_size * 2, self.small_h_size), nn.Tanh(),
        #                               nn.Linear(self.small_h_size, self.small_out_size)
        #                               )
        #
        # self.intr_tm_ann = nn.Sequential(nn.Linear(self.small_in_size, self.small_h_size * 2), nn.Tanh(),
        #                                  nn.Linear(self.small_h_size * 2, self.small_h_size), nn.Tanh(),
        #                                  nn.Linear(self.small_h_size, self.small_out_size)
        #                                  )

        # self.in_size = 2 * self.small_out_size + 1  # first_n_pkts_list, flow_duration, intr_time_list
        self.in_size = 60
        self.h_size = 5
        # self.out_size = 1  # number of label, one-hot coding
        # self.classify_ann = nn.Sequential(nn.Linear(self.in_size, self.h_size * 2), nn.Tanh(),
        #                                   nn.Linear(self.h_size * 2, self.h_size), nn.Tanh(),
        #                                   nn.Linear(self.h_size, self.out_size, nn.Softmax())
        #                                   )

        # For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.
        #
        # If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.

        # self.conv1 = nn.Conv2d(1, 6, (5, 1), stride=1)
        # self.conv2 = nn.Conv2d(6, 16, (5, 1), stride=1)
        # # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(16 * 1 * 5, 120)
        self.fc1 = nn.Linear(60, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

        # self.classify_ann = nn.Sequential(self.conv1, nn.Tanh(),
        #                                   self.conv2, nn.Tanh(),
        #                                   self.fc1,nn.Tanh(),
        #                                   self.fc2,nn.Tanh(),
        #                                   self.fc3
        #                                   )

        print('---------- Networks architecture -------------')
        # print_network('pkts_ann:', self.pkts_ann)
        # print_network('intr_tm_ann:', self.intr_tm_ann)
        # print_network('classify_ann:', self.classify_ann)
        # print('-----------------------------------------------')

        self.criterion = nn.MSELoss(size_average=False)
        # self.criterion = nn.MultiLabelMarginLoss()
        self.learning_rate = 1e-4
        # self.optimizer = torch.optim.Adam(self.proposed_algorithms.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.Adam([self.pkts_ann, self.intr_tm_ann, self.classify_ann], lr=self.d_learning_rate,
        #                             betas=(0.5, 0.9))
        # params = list(self.pkts_ann.parameters()) + list(self.intr_tm_ann.parameters()) + list(
        #     self.classify_ann.parameters())
        params = list(self.parameters())
        self.optimizer = optim.Adam(params, lr=self.learning_rate, betas=(0.5, 0.9))

    # def forward(self, X):
    #
    #     pkts_x = X[:, 0:self.first_n_pkts]
    #     flow_dur = X[:, self.first_n_pkts]
    #     intr_x = X[:, self.first_n_pkts + 1:2 * self.first_n_pkts + 1]
    #
    #     pkts_outputs = self.pkts_ann(pkts_x)
    #     # flow_dur = flow_dur
    #     intr_outputs = self.intr_tm_ann(intr_x)
    #
    #     new_X = []
    #     for i in range(len(X)):
    #         lst_tmp = []
    #         lst_tmp.append(flow_dur[i].input_data.tolist())
    #         lst_tmp.extend(pkts_outputs[i].input_data.tolist())
    #         lst_tmp.extend(intr_outputs[i].input_data.tolist())
    #         new_X.append(lst_tmp)
    #     # X = [pkts_outputs, flow_dur, intr_outputs]
    #     new_X = torch.Tensor(new_X)
    #     y_preds = self.classify_ann(new_X)
    #     # _, y_preds=y_preds.input_data.max(dim=1) # get max value of each row
    #
    #     return y_preds

    def forward(self, x):

        # y_preds = self.classify_ann(X)
        # # _, y_preds=y_preds.input_data.max(dim=1) # get max value of each row
        # Max pooling over a (2, 2) window
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 1))
        # # If the size is a square you can only specify a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 1))
        # x =F.relu(self.conv1(x))
        # x=F.relu(self.conv2(x))
        # x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

        # return y_preds

    def train(self, training_set):
        self.train_hist = {}
        self.train_hist['acc'] = []
        self.train_hist['loss'] = []
        self.test_hist = {}
        self.test_hist['acc'] = []
        self.test_hist['loss'] = []

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
                # b_y = Variable(b_y.long())
                y_preds = self.forward(b_x)
                loss = self.criterion(y_preds, b_y)  # net_outs, y_real(targets)

                self.optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                self.optimizer.step()  # apply gradients

                self.train_hist['loss'].append(loss.data.tolist())

                X_train_set = torch.Tensor(X_train)
                Y_preds = ann.predict(X_train_set)
                acc = ann.evaluate(y_train, Y_preds)
                # print('testing accuracy:', acc)
                self.train_hist['acc'].append(acc)

                X_test_set = torch.Tensor(X_test)
                Y_preds = ann.predict(X_test_set)
                acc = ann.evaluate(y_test, Y_preds)
                # print('testing accuracy:', acc)
                self.test_hist['acc'].append(acc)

                # self.test_hist['loss'].append(loss.input_data.tolist())

                if step % 100 == 0:
                    print('epoch = %d, loss = %f' % (epoch, loss.data.tolist()))

    def predict(self, X):
        y_preds = self.forward(X)
        _, y_ = y_preds.data.max(dim=1, keepdim=False)  # return max_value as predicted value

        # y_preds=y_preds.input_data.tolist()
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


def one_hot(ids, out_tensor):
    """
    ids: (list, ndarray) shape:[batch_size]
    out_tensor:FloatTensor shape:[batch_size, depth]
    """
    if not isinstance(ids, (list, np.ndarray)):
        raise ValueError("ids must be 1-D list or array")
    ids = torch.LongTensor(ids)
    out_tensor.zero_()
    return out_tensor.scatter_(dim=1, index=ids, value=1)


# out_tensor.scatter_(1, ids, 1.0)


def one_hot_sklearn(label_integer):
    label_integer = np.asarray(label_integer, dtype=int)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = label_integer.reshape(len(label_integer), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return np.array(onehot_encoded, dtype=int)


if __name__ == '__main__':
    torch.manual_seed(1)  # reproducible

    # input_file = '../results/AUDIO_first_n_pkts_10_all_in_one_file.txt'
    input_file = '../input_data/data_split_train_v2_711/train_1pkt_images_merged.csv'
    X, Y = load_data(input_file)
    X = normalize_data(np.asarray(X, dtype=float), range_value=[-1, 1], eps=1e-5)
    Y = change_label(Y)
    X_train, X_test, y_train, y_test = achieve_train_test_data(X, Y, train_size=0.9, shuffle=True)

    input_file = '../input_data/data_split_train_v2_711/train_1pkt_images_merged.csv'
    # print('input_file:', input_file)
    # X, Y = load_sequence_data(input_file)
    # print('Y :', Counter(Y))
    # X_train, X_test, y_train, y_test = achieve_train_test_data(X, Y, train_size=0.7, shuffle=True)
    print(
        'X_train : %d, y_train : %d, label : %s' % (len(X_train), len(y_train), dict(sorted(Counter(y_train).items()))))
    # print('y_train : %s\ny_test  : %s'%(Counter(y_train), Counter(y_test)))
    print('X_test  : %d, y_test  : %d, label : %s' % (len(X_test), len(y_test), dict(sorted(Counter(y_test).items()))))

    ann = ANN(BATCH_SIZE=16, first_n_pkts=1, epochs=1000, num_class=len(Counter(y_train)))
    # training_set = Data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))  # X, Y
    one_hot_y_train = one_hot_sklearn(y_train)
    training_set = Data.TensorDataset(torch.Tensor(X_train), torch.Tensor(one_hot_y_train))  # X, Y
    ann.train(training_set)

    show_figure(ann.train_hist['loss'])
    show_figure(ann.train_hist['acc'])
    show_figure(ann.test_hist['acc'])

    X_train_set = torch.Tensor(X_train)
    Y_preds = ann.predict(X_train_set)
    print(Counter(Y_preds))
    acc = ann.evaluate(y_train, Y_preds)
    print('training accuracy:', acc)

    X_test_set = torch.Tensor(X_test)
    Y_preds = ann.predict(X_test_set)
    print(Counter(Y_preds))
    acc = ann.evaluate(y_test, Y_preds)
    print('testing accuracy:', acc)
#
#     # Hyper Parameters
#     EPOCH = 1  # train the training input_data n times, to save time, we just train 1 epoch
#     BATCH_SIZE = 50
#     LR = 0.001  # learning rate
#     DOWNLOAD_MNIST = False
#
# # Mnist digits dataset
# if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
#     # not mnist dir or mnist is empyt dir
#     DOWNLOAD_MNIST = True
#

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
#             nn.Conv2d(
#                 in_channels=1,              # input height
#                 out_channels=16,            # n_filters
#                 kernel_size=5,              # filter size
#                 stride=1,                   # filter movement/step
#                 padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
#             ),                              # output shape (16, 28, 28)
#             nn.ReLU(),                      # activation
#             nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
#         )
#         self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
#             nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
#             nn.ReLU(),                      # activation
#             nn.MaxPool2d(2),                # output shape (32, 7, 7)
#         )
#         self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
#         output = self.out(x)
#         return output, x    # return x for visualization
#
#
# cnn = CNN()
# print(cnn)  # net architecture
#
# optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
# loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
#
# # following function (plot_with_labels) is for visualization, can be ignored if not interested
# from matplotlib import cm
# try: from sklearn.manifold import TSNE; HAS_SK = True
# except: HAS_SK = False; print('Please install sklearn for layer visualization')
# def plot_with_labels(lowDWeights, labels):
#     plt.cla()
#     X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
#     for x, y, s in zip(X, Y, labels):
#         c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
#     plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)
#
# plt.ion()
# # training and testing
# for epoch in range(EPOCH):
#     for step, (b_x, b_y) in enumerate(train_loader):   # gives batch input_data, normalize x when iterate train_loader
#
#         output = cnn(b_x)[0]               # cnn output
#         loss = loss_func(output, b_y)   # cross entropy loss
#         optimizer.zero_grad()           # clear gradients for this training step
#         loss.backward()                 # backpropagation, compute gradients
#         optimizer.step()                # apply gradients
#
#         if step % 50 == 0:
#             test_output, last_layer = cnn(test_x)
#             pred_y = torch.max(test_output, 1)[1].input_data.squeeze()
#             accuracy = float(sum(pred_y == test_y)) / float(test_y.size(0))
#             print('Epoch: ', epoch, '| train loss: %.4f' % loss.input_data.numpy(), '| test accuracy: %.2f' % accuracy)
#             if HAS_SK:
#                 # Visualization of trained flatten layer (T-SNE)
#                 tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#                 plot_only = 500
#                 low_dim_embs = tsne.fit_transform(last_layer.input_data.numpy()[:plot_only, :])
#                 labels = test_y.numpy()[:plot_only]
#                 plot_with_labels(low_dim_embs, labels)
# plt.ioff()
#
# # print 10 predictions from test input_data
# test_output, _ = cnn(test_x[:10])
# pred_y = torch.max(test_output, 1)[1].input_data.numpy().squeeze()
# print(pred_y, 'prediction number')
# print(test_y[:10].numpy(), 'real number')
