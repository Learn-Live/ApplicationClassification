# -*- coding: utf-8 -*-
r"""

"""
import argparse
import copy
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from matplotlib.animation import FuncAnimation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset

from sys_path_export import *  # it is no need to do in IDE environment, however, it must be done in shell/command environment

# # matplotlib.use("Agg")
# import matplotlib.animation as manimation
from proposed_algorithms.numpy_load import load_npy_data


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


class TrafficDataset(Dataset):

    def __init__(self, X, y, transform=None, normalization_flg=False):
        self.X = X
        self.y = y
        cnt = 0
        # with open(input_file, 'r') as fid_in:
        #     line = fid_in.readline()
        #     while line:
        #         line_arr = line.split(',')
        #         value = list(map(lambda x: float(x), line_arr[:-1]))
        #         self.X.append(value)
        #         self.y.append(float(line_arr[-1].strip()))
        #         line = fid_in.readline()
        #         cnt += 1
        if normalization_flg:
            self.X = normalize_data(np.asarray(self.X, dtype=float), range_value=[-1, 1], eps=1e-5)
            # with open(input_file + '_normalized.csv', 'w') as fid_out:
            #     for i in range(self.X.shape[0]):
            #         # print('i', i.input_data.tolist())
            #         tmp = [str(j) for j in self.X[i]]
            #         fid_out.write(','.join(tmp) + ',' + str(variables_n_data_types_issues(self.y[i])) + '\n')

        self.transform = transform

    def __getitem__(self, index):

        value_x = self.X[index]
        value_y = self.y[index]
        if self.transform:
            value_x = self.transform(value_x)

        value_x = torch.from_numpy(np.asarray(value_x)).double()
        value_y = torch.from_numpy(np.asarray(value_y)).double()

        # X_train, X_test, y_train, y_test = train_test_split(value_x, value_y, train_size=0.7, shuffle=True)
        return value_x, value_y  # Dataset.__getitem__() should return a single sample and label, not the whole dataset.
        # return value_x.view([-1,1,-1,1]), value_y

    def __len__(self):
        return len(self.X)


def print_network(describe_str, net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(describe_str, net)
    print('Total number of parameters: %d' % num_params)


def generated_train_set(num):
    X = []
    y = []
    for i in range(num):
        yi = 0
        if i % 2 == 0:
            yi = 1
        rnd = np.random.random()
        rnd2 = np.random.random()
        X.append([1000 * rnd, i * 1 * rnd2])
        y.append(yi)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    return TrafficDataset(X, y, normalization_flg=False)


class PrintLayer(nn.Module):
    def __init__(self, idx_layer):
        super(PrintLayer, self).__init__()
        self.idx_layer = idx_layer

    def forward(self, x):
        # Do your print / debug stuff here
        print('print_%sth_layer (batch_size x out_dim)=%s' % (self.idx_layer, x.shape))
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class NeuralNetworkDemo():
    r"""
        Visualize neural network parameters

            print the weights and bias values
    """

    def __init__(self, in_dim, out_dim=10, epochs=10, display_flg=False):
        self.display_flg = display_flg

        self.in_dim = in_dim
        self.h_dim = 30
        self.out_dim = out_dim

        self.epochs = epochs

        # # method 1: network structure (recommend), however it is not easy to print values in each layer
        # in_lay = nn.Linear(self.in_dim, self.h_dim * 20, bias=True)  # class_issues initialization
        # hid_lay = nn.Linear(self.h_dim * 20, self.h_dim * 10, bias=True)
        # hid_lay_2 = nn.Linear(self.h_dim * 10, self.h_dim * 10, bias=False)
        # hid_lay_3 = nn.Linear(self.h_dim * 10, self.h_dim * 20, bias=False)
        # out_lay = nn.Linear(self.h_dim * 20, self.out_dim, bias=True)
        # # self.net = nn.Sequential(
        # #                          in_lay,
        # #                          nn.Sigmoid(),
        # #                          hid_lay,
        # #                          nn.LeakyReLU(),
        # #                          hid_lay_2,
        # #                          nn.LeakyReLU(),
        # #                          out_lay
        # #                          )
        # # refer to : https://discuss.pytorch.org/t/how-do-i-print-output-of-each-layer-in-sequential/5773/4
        # self.net = nn.Sequential(#PrintLayer(idx_layer=0),  # Add Print layer for debug
        #                          in_lay,
        #                          #PrintLayer(idx_layer=1),  # Add Print layer for debug
        #                          nn.LeakyReLU(),
        #                          hid_lay,
        #                          #PrintLayer(idx_layer=2),  # Add Print layer for debug
        #                          nn.LeakyReLU(),
        #                         #  hid_lay_2,
        #                         # nn.LeakyReLU(),
        #                         # hid_lay_2,
        #                         # nn.LeakyReLU(),
        #                         # hid_lay_2,
        #                          nn.LeakyReLU(),
        #                          hid_lay_3,
        #                          #PrintLayer(idx_layer=3),  # Add Print layer for debug
        #                          nn.LeakyReLU(),
        #                          out_lay,
        #                          #PrintLayer(idx_layer='out'),  # Add Print layer for debug
        #                          )

        # # method 2 : it is not easy to use , however it is easy to print values in each layer.
        # class NN(nn.Module):
        #     def __init__(self, in_dim, h_dim, out_dim):
        #         super(NN, self).__init__()
        #         self.in_dim = in_dim
        #         self.h_dim = h_dim
        #         self.out_dim = out_dim
        #         self.in_lay = nn.Linear(self.in_dim, self.h_dim * 20, bias=True)  # class_issues initialization
        #         self.hid_lay = nn.Linear(self.h_dim * 20, self.h_dim * 10, bias=True)
        #         self.hid_lay_2 = nn.Linear(self.h_dim * 10, self.h_dim * 20, bias=False)
        #         self.out_lay = nn.Linear(self.h_dim * 20, self.out_dim, bias=True)
        #
        #     def forward(self, X):
        #         z1 = self.in_lay(X)
        #         # a1=nn.Sigmoid(z1)
        #         a1 = F.leaky_relu(z1)
        #         z2 = self.hid_lay(a1)
        #         a2 = F.leaky_relu(z2)
        #         z3 = self.hid_lay_2(a2)
        #         a3 = F.leaky_relu(z3)
        #         z4 = self.out_lay(a3)
        #         out = F.softmax(z4)
        #
        #         return out

        # self.net = NN(self.in_dim, self.h_dim, self.out_dim)

        # Convolutional neural network (two convolutional layers)
        class ConvNet(nn.Module):
            def __init__(self, num_classes=10):
                super(ConvNet, self).__init__()
                # 1 input image channel, 6 output channels, 5x1 square convolution
                self.layer1 = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=(5, 1), stride=1),
                    # nn.BatchNorm2d(16),
                    nn.Tanh(),
                    # nn.MaxPool2d(kernel_size=(2,1), stride=2)
                )
                self.layer2 = nn.Sequential(
                    nn.Conv2d(16, 32, kernel_size=(5, 1), stride=1, padding=0),
                    # nn.BatchNorm2d(32),
                    nn.Tanh(),
                    # nn.MaxPool2d(kernel_size=(2,1), stride=2)
                )
                self.fc1 = nn.Linear(32 * 1992 * 1, 1000*1)
                self.fc2 = nn.Linear(1000 * 1, num_classes)

            def forward(self, x):
                out = self.layer1(x)
                out = self.layer2(out)
                out = out.reshape(out.size(0), -1)
                out = self.fc1(out)
                out = self.fc2(out)
                return out

        self.net = ConvNet(self.out_dim).to(device)

        ## evaluation standards
        ## self.criterion = nn.MSELoss()  # class_issues initialization
        self.criterion = nn.CrossEntropyLoss()  # class_issues initialization

        # optimizer
        self.optim = optim.Adam(self.net.parameters(), lr=1e-4, betas=(0.9, 0.99))
        print(callable(self.optim))

        if display_flg:
            # print network architecture
            print_network('demo', self.net)
            print_net_parameters(self.net, OrderedDict(), title='Initialization parameters')

    def forward(self, X):
        """
            more flexible and efficient than Sequential()
        :param X:
        :return:
        """
        # out = self.net.forward(X)
        out = self.net(X)
        out = F.softmax(out)
        return out

    # def forward_sequential(self, X):
    #     o1 = self.net(X)
    #
    #     return o1

    def train(self, train_set, train_set_tuple, test_set):
        print('training')
        # X,y = train_set
        # train_set = (torch.from_numpy(X).double(), torch.from_numpy(y).double())
        self.batch_size = 50
        train_loader = Data.DataLoader(train_set, self.batch_size, shuffle=True, num_workers=4)
        all_params_order_dict = OrderedDict()
        ith_layer_out_dict = OrderedDict()
        learn_rate_lst = []

        loss_lst = []
        test_acc_lst = []
        train_acc_lst = []
        for epoch in range(self.epochs):
            param_order_dict = OrderedDict()
            loss_tmp = torch.Tensor([0.0])
            for batch_idx, (b_x, b_y) in enumerate(train_loader):
                # b_x = b_x.view([b_x.shape[0], -1]).float()
                b_x = b_x.view([b_x.shape[0], 1, -1, 1]).float()
                b_y = b_y.view(b_y.shape[0], 1).long()
                b_y = b_y.squeeze_()

                self.optim.zero_grad()
                b_y_preds = self.forward(b_x)
                loss = self.criterion(b_y_preds, b_y)
                lr = self.optim.param_groups[0]['lr']
                loss.backward()
                self.optim.step()

                # for graphing purposes
                learn_rate_lst.append(lr)
                loss_tmp += loss.data
                # # print the current status of training
                # if (batch_idx % 100 == 0):
                #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         epoch, batch_idx * len(b_x), len(train_loader.dataset),
                #                100. * batch_idx / len(train_loader), loss.input_data[0]))
                if batch_idx == 0:
                    print('%d/%d, batch_ith = %d, loss=%f, lr=%s' % (epoch, self.epochs, batch_idx, loss.data, lr))
                # for idx, param in enumerate(self.net.parameters()):
                for name, param in self.net.named_parameters():
                    # print(name, param)  # even is weigh and bias, odd is activation function, it's no parameters.
                    if name not in param_order_dict.keys():
                        param_order_dict[name] = copy.deepcopy(param.data.numpy())  # numpy arrary
                    else:
                        # param_order_dict[name].append(copy.deepcopy(np.reshape(param.input_data.numpy(), (-1, 1))))
                        param_order_dict[name] += copy.deepcopy(param.data.numpy())  # numpy arrary
            loss_lst.append(loss_tmp.data / len(train_loader))
            if epoch not in all_params_order_dict.keys():  # key = epoch, value =param_order_dict
                # average parameters
                all_params_order_dict[epoch] = {key: value / len(train_loader) for key, value in
                                                param_order_dict.items()}

                # evaluation on train set
                X_train, y_train = train_set_tuple
                b_x=torch.Tensor(X_train)
                b_x = b_x.view([b_x.shape[0], 1, -1, 1]).float()
                y_preds = self.forward(b_x)
                y_preds = torch.argmax(y_preds, dim=1).numpy()  # get argmax value, predict label
                print(confusion_matrix(y_train, y_preds))
                train_acc = metrics.accuracy_score(y_train, y_preds)
                print("train acc", train_acc)
                train_acc_lst.append(train_acc)

                # evaluation on Test set
                X_test, y_test = test_set
                b_x = torch.Tensor(X_test)
                b_x = b_x.view([b_x.shape[0], 1, -1, 1]).float()
                y_preds = self.forward(b_x)
                y_preds = torch.argmax(y_preds, dim=1).numpy()  # get argmax value, predict label
                print(confusion_matrix(y_test, y_preds))
                test_acc = metrics.accuracy_score(y_test, y_preds)
                print("test acc", test_acc)
                test_acc_lst.append(test_acc)

        save_data(train_acc_lst, 'train_acc_lst.txt')
        save_data(test_acc_lst, 'test_acc_lst.txt')

        if self.display_flg:
            plot_data(loss_lst, x_label='epochs', y_label='loss', title='training model')
            plot_data(train_acc_lst, x_label='epochs', y_label='train_acc', title='training model')
            plot_data(test_acc_lst, x_label='epochs', y_label='test_acc', title='testing model')
            # live_plot_params(self.net, all_params_order_dict, output_file='dynamic.mp4')

            # print_net_parameters(self.net, param_order_dict,
            #                      title='All parameters (weights and bias) from \n begin to finish in training process phase.')
            #
            # print_net_parameters(self.net, OrderedDict(), title='Final parameters')


def save_data(data, out_file='output.txt'):
    with open(out_file, 'w') as out_hdl:
        for v_tmp in data:
            out_hdl.write(str(v_tmp) + '\n')

    return out_file


def plot_data(data, x_label, y_label, title=''):
    r"""

    :param data:
    :param x_label:
    :param y_label:
    :param title:
    :return:
    """
    # recommend to use, plt.subplots() default parameter is (111)
    fig, ax = plt.subplots()  # combine plt.figure and fig.add_subplots(111)
    ax.plot(data)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    # ax.set_ylabel()
    plt.show()


def live_plot_params(net, all_params_order_dict, output_file='dynamic.mp4'):
    r"""
        save the change of parameters in each epcoh to .mp4 file

        must install ffmpeg, then pip3 install ffmpeg
        Note:
            pycharm cannot show animation. so it needs to save animation to local file.

    :param net:  neural network based on pytorch
    :param all_params_order_dict:
    :param output_file:
    :return:
    """
    num_figs = len(all_params_order_dict[0]) // 2 + 1  # number of layers in nn
    fig, axes = plt.subplots(nrows=num_figs, ncols=2)  # create fig and add subplots (axes) into it.
    ax_lst = []

    def update(frame_data):
        ith_epoch, ith_param_order_dict = frame_data  # dictionary
        fontsize = 7
        for ax_i, (key, value) in zip(axes.flatten(), ith_param_order_dict.items()):
            ax_i.clear()  # clear the previous input_data, then redraw the new input_data.
            num_bins = value.size // 2
            if num_bins < 10:
                num_bins = 10
            print('epoch=%s, key=%s' % (ith_epoch, key))
            y_tmp = np.reshape(np.asarray(value, dtype=float), (-1, 1))
            # n, bins, patches = ax_i.hist(np.reshape(np.asarray(value, dtype=float), (-1, 1)), num_bins,
            #                              facecolor='blue', alpha=0.5)
            ax_i.scatter(range(value.size), y_tmp, c=y_tmp, s=2)
            ax_i.set_xlabel('Values', fontsize=fontsize)
            ax_i.set_ylabel('Frequency', fontsize=fontsize)
            # ax_i.set_xticks(range(6))
            # ax_i.set_xticklabels([str(x) + "foo" for x in range(6)], rotation=45, fontsize=fontsize)
            # ax_i.set_xticklabels(ax_i.get_xticks(),fontsize=fontsize)
            for label in (ax_i.get_xticklabels() + ax_i.get_yticklabels()):
                label.set_fontname('Arial')
                label.set_fontsize(fontsize)
            # ax_i.set_xlim(-1,1)
            # ax_i.set_ylim(0,value.size)
            ax_i.set_title('%s:(%s^T)' % (key, value.shape), fontsize=fontsize)  # paramter_name and shape
        fig.suptitle('epoch:%d' % ith_epoch)

        return ax_lst

    def new_data():
        for ith_epoch, ith_param_order_dict in all_params_order_dict.items():
            print('epoch(%d)/epochs(%d)' % (ith_epoch, len(all_params_order_dict.keys())))
            yield ith_epoch, ith_param_order_dict

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # rect: tuple (left, bottom, right, top), optional
    # tight_layout(pad=0.05, w_pad=0.001, h_pad=2.0)
    # fig.subplots_adjust(top=0.88)
    anim = FuncAnimation(fig, update, frames=new_data, repeat=False, interval=1000,
                         blit=False)  # interval : ms
    anim.save(output_file, writer='ffmpeg', fps=None, dpi=400)
    plt.show()


def print_net_parameters(net, param_order_dict=OrderedDict(), title=''):
    r"""

    :param net:
    :param param_order_dict:
    :param title:
    :return:
    """

    if param_order_dict == {}:
        # for idx, param in enumerate(self.net.parameters()):
        for name, param in net.named_parameters():
            print(name, param)  # even is weigh and bias, odd is activation function, it's no parameters.
            if name not in param_order_dict.keys():
                param_order_dict[name] = copy.deepcopy(np.reshape(param.data.numpy(), (-1, 1)))
            else:
                print('error:', name)

    num_figs = len(param_order_dict.keys()) // 2 + 1
    print('subplots:(%dx%d):' % (num_figs, num_figs))
    print(title)
    fig, axes = plt.subplots(nrows=num_figs, ncols=2)
    fontsize = 10
    # plt.suptitle(title, fontsize=8)
    x_label = 'Values'
    y_label = 'Frequency'
    for ith, (ax_i, (name, param)) in enumerate(zip(axes.flatten(), net.named_parameters())):
        # for ith, (name, param) in enumerate(net.named_parameters()):
        print('subplot_%dth' % (ith + 1))
        num_bins = 10
        x_tmp = np.reshape(np.asarray(param_order_dict[name], dtype=float), (-1, 1))
        n, bins, patches = ax_i.hist(x_tmp, num_bins, facecolor='blue', alpha=0.5)
        ax_i.set_xlabel('Values', fontsize=fontsize)
        ax_i.set_ylabel('Frequency', fontsize=fontsize)
        ax_i.set_title('%s:(%s^T)' % (name, param.data.numpy().shape), fontsize=fontsize)  # paramter_name and shape
    fig.suptitle(title, fontsize=fontsize)
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])  # rect: tuple (left, bottom, right, top), optional
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.88)
    plt.show()


# TODO list : plot_issues func.

# def show_figures(D_loss, G_loss):
#     import matplotlib.pyplot as plt
#     fig, axes=plt.subplots(111)
#
#     plt.plot_issues(D_loss, 'r', alpha=0.5, label='D_loss of real and fake sample')
#     plt.plot_issues(G_loss, 'g', alpha=0.5, label='D_loss of G generated fake sample')
#     plt.legend(loc='upper right')
#     plt.title('D\'s loss of real and fake sample.')
#     plt.show()


def parse_params():
    parser = argparse.ArgumentParser(prog='nn_application')
    parser.add_argument('-i', '--input_file', type=str, dest='input_file',
                        help='\'normal_files\'',
                        default='../input_data/trdata-8000B.txt',
                        required=True)  # '-i' short name, '--input_dir' full name
    parser.add_argument('-e', '--epochs', dest='epochs', help="num of epochs", default=10)
    parser.add_argument('-o', '--out_dir', dest='out_dir', help="the output information of this scripts",
                        default='../log')
    args = vars(parser.parse_args())

    return args


def load_data_and_plot(input_file):
    data = []
    with open(input_file, 'r') as in_hdl:
        line = in_hdl.readline()
        while line:
            data.append(float(line.strip()))
            line = in_hdl.readline()
    plot_data(data)


def app_main(input_file, epochs, out_dir='../log'):
    """

    :param input_file:
    :param epochs:
    :param out_dir:
    :return:
    """
    # torch.manual_seed(1)
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    st = time.time()
    print('It starts at ', start_time)

    # # train_set = generated_train_set(100)
    # input_file = '../input_data/trdata-8000B.npy'
    session_size = 2000
    print(f'session_size:{session_size}')
    X, y = load_npy_data(input_file, session_size,norm_flg=True)
    test_percent = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, random_state=42)
    print(f'train_test_ratio:[{1-test_percent}:{test_percent}]')

    train_set = TrafficDataset(X_train, y_train, normalization_flg=False)
    # test_set = TrafficDataset(X_test, y_test, normalization_flg=False)

    nn_demo = NeuralNetworkDemo(in_dim=session_size, epochs=epochs, display_flg=True)
    nn_demo.train(train_set, (X_train, y_train), (X_test, y_test))

    end_time = time.strftime('%Y-%h-%d %H:%M:%S', time.localtime())
    print('\nIt ends at ', end_time)
    print('All takes %.4f s' % (time.time() - st))


if __name__ == '__main__':
    args = parse_params()
    # print(args['input_file'])
    input_file = args['input_file']
    epochs = eval(args['epochs'])
    out_dir = args['out_dir']
    print('args:%s\n' % args)
    app_main(input_file, epochs, out_dir='../log')
