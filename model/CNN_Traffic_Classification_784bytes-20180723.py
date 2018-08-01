# -*- coding:utf-8 -*-
"""

    refer: https://raw.githubusercontent.com/yunjey/pytorch-tutorial/master/tutorials/02-intermediate/convolutional_neural_network/main.py
"""
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.autograd import Variable
# Device configuration
from torch.utils.data.sampler import SubsetRandomSampler

from preprocess import idx_reader
from preprocess.TrafficDataset import TrafficDataset, split_train_test

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    # def __init__(self, num_classes=10, num_features=60):
    #
    #     # self.train_loader= train_loader
    #     super(ConvNet, self).__init__()
    #     # 1 input image channel, 6 output channels, 5x1 square convolution
    #     self.layer1 = nn.Sequential(
    #         nn.Conv2d(1, 256, kernel_size=(10, 1), stride=1,padding=(2,0)),
    #         # nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width
    #         # nn.BatchNorm2d(3),
    #         nn.Tanh(),
    #         # nn.MaxPool2d(kernel_size=(2,1), stride=2)
    #     )
    #     self.layer2 = nn.Sequential(
    #         nn.Conv2d(256, 128, kernel_size=(5, 1), stride=3, padding=(2,0)),
    #         # nn.BatchNorm2d(2),
    #         nn.ReLU(),
    #         # nn.MaxPool2d(kernel_size=(2,1), stride=2)
    #     )
    #     self.fc1 = nn.Linear(128 * ((((num_features - 10+2*2)//1 +1) - 5+2*2)//3 +1) ,           # ouput = (input-filter+2*padding)/stride +1
    #                         num_classes*20)  # (1, 16, 60*i +i-1-(5-1),1) -> (16, 32, 60*i +i-1-(5-1) -(3-1),1)
    #
    #     self.fc = nn.Linear(num_classes*20,
    #                         num_classes)  # (1, 16, 60*i +i-1-(5-1),1) -> (16, 32, 60*i +i-1-(5-1) -(3-1),1)
    #
    #     # Loss and optimizer
    #     self.criterion = nn.CrossEntropyLoss()
    #     # self.criterion=nn.NLLLoss()
    #     self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9,0.999))
    #     # self.optimizer = torch.optim.Adadelta(self.parameters(), lr=learning_rate)
    def __init__(self, num_classes=10, num_features=60):

        # self.train_loader= train_loader
        super(ConvNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x1 square convolution
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 1), stride=1, padding=(2, 0)),
            # nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width
            # nn.BatchNorm2d(3),
            nn.Tanh(),
            # nn.MaxPool2d(kernel_size=(2,1), stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(1, 1), stride=1, padding=(2, 0)),
            # nn.BatchNorm2d(2),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2,1), stride=2)
        )
        self.fc1 = nn.Linear(32 * ((((num_features - 1 + 2 * 2) // 1 + 1) - 1 + 2 * 2) // 1 + 1),
                             # ouput = (input-filter+2*padding)/stride +1
                             num_classes * 20)  # (1, 16, 60*i +i-1-(5-1),1) -> (16, 32, 60*i +i-1-(5-1) -(3-1),1)

        self.fc = nn.Linear(num_classes * 20,
                            num_classes)  # (1, 16, 60*i +i-1-(5-1),1) -> (16, 32, 60*i +i-1-(5-1) -(3-1),1)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion=nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        # self.optimizer = torch.optim.Adadelta(self.parameters(), lr=learning_rate)

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
            # while (cnt - 1) * num_features + (cnt - 1) < len(sentence_i):
            #     tmp_lst.append(
            #         sentence_i[(cnt - 1) * num_features + (cnt - 1): cnt * num_features + (cnt - 1)].data.tolist()[
            #         :num_features])
            #     # t = (cnt - 1) * 60 + (cnt - 1)
            #     if cnt == first_n_pkts:
            #         break
            #     cnt += 1
            # tmp_lst=sentence_i.data.tolist()[:first_n_pkts*num_features]
            # tmp_lst = torch.from_numpy(np.array(tmp_lst))
            tmp_lst = sentence_i[:first_n_pkts * num_features]
            new_sentences.append(tmp_lst)

        new_sentences = torch.stack(new_sentences)

        return new_sentences

    def forward(self, x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        out = layer2_out.reshape(layer2_out.size(0), -1)
        out = self.fc1(out)
        out = self.fc(out)
        # out= nn.Softmax(out)
        return out, layer2_out, layer1_out

    def l1_penalty(self, var):
        return torch.abs(var).sum()

    def l2_penalty(self, var):
        return torch.sqrt(torch.pow(var, 2).sum())

    def upsampling_data(self, b_x, b_y):
        """
            to make proportition as equal
        :param b_x:
        :param b_y:
        :return:
        """
        b_y_items = Counter(b_y.data.tolist())
        max_label, max_label_num = sorted(b_y_items.items(), key=lambda x: x[1], reverse=False)[
            -1]  # get max value in dictionary

        # c =  list(zip(b_x, b_y))
        # a=[]
        # for key in b_y_items.keys():
        #     for b_x_i, b_y_i in zip(b_x, b_y):
        #         if key == b_y_i.data.tolist():
        #             a[key].append(b_x_i)

        new_b_x = []
        new_b_y = []
        for key in b_y_items.keys():
            b_x_tmp = list(b_x.data.numpy()[np.asarray(b_y.data.tolist()) == key])
            b_x_tmp = list(map(lambda x: torch.Tensor(x), b_x_tmp))
            b_y_tmp = b_y.data.numpy()[np.asarray(b_y.data.tolist()) == key]
            b_y_tmp = [torch.Tensor([key]) for _ in range(len(b_y_tmp))]
            new_b_x.extend(b_x_tmp)
            new_b_y.extend(b_y_tmp)
            if b_y_items[key] != max_label_num:
                for c in range((max_label_num - len(b_y_tmp)) // len(b_y_tmp)):
                    b_x_tmp = random.sample(list(b_x.data.numpy()[np.asarray(b_y.data.tolist()) == key]), len(b_y_tmp))
                    b_x_tmp = list(map(lambda x: torch.Tensor(x), b_x_tmp))
                    b_y_tmp = [torch.Tensor([key]) for _ in range(len(b_y_tmp))]
                    new_b_x.extend(b_x_tmp)
                    new_b_y.extend(b_y_tmp)
                if max_label_num % len(b_y_tmp):
                    b_x_tmp = random.sample(list(b_x.data.numpy()[np.asarray(b_y.data.tolist()) == key]),
                                            max_label_num % len(b_y_tmp))
                    b_x_tmp = list(map(lambda x: torch.Tensor(x), b_x_tmp))
                    b_y_tmp = [torch.Tensor([key]) for _ in range(max_label_num % len(b_y_tmp))]
                    new_b_x.extend(b_x_tmp)
                    new_b_y.extend(b_y_tmp)

        new_b_x = torch.stack(new_b_x, dim=0)
        new_b_y = torch.stack(new_b_y, dim=-1)[0]
        # c = list(zip(new_b_x, new_b_y))
        # np.random.shuffle(c)
        # new_b_x, new_b_y = list(zip(*c))

        return new_b_x, new_b_y


    def run_train(self, train_loader, mode=True):
        # Train the model
        self.results = {}
        self.results['train_acc'] = []
        self.results['train_loss'] = []

        self.results['test_acc'] = []
        self.results['test_loss'] = []

        total_step = len(train_loader)
        for epoch in range(EPOCHES):
            for i, (b_x, b_y) in enumerate(train_loader):
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                # b_x=self.achieve_sentence(b_x,first_n_pkts)
                b_x = b_x[:, :first_n_pkts * num_features]
                b_x, b_y = self.upsampling_data(b_x, b_y)
                b_x = b_x.view([b_x.shape[0], 1, -1, 1])
                b_x = Variable(b_x).float()
                b_y = Variable(b_y).type(torch.LongTensor)
                # Forward pass
                # b_y_preds = model(b_x)
                b_y_preds, layer2_out, layer1_out = self.forward(b_x)
                # l1_regularization = self.l1_penalty(layer1_out)
                # l1_regularization = self.l1_penalty(layer2_out)
                # l1_regularization += self.l1_penalty(b_y_preds)
                # # l2_regularization = 1e-2 * self.l2_penalty(layer2_out)
                # l1_regularization= self.l1_penalty(self.parameters())
                l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
                for W in self.parameters():
                    # l2_reg = l2_reg+ W.norm(1)
                    l2_reg = l2_reg + W.norm(2) ** 2
                    # l2_reg +=  torch.pow(W, 2).sum()
                    # print(W.data.tolist())
                #
                loss = self.criterion(b_y_preds, b_y) + 0 * l2_reg

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # if (i + 1) % 50 == 0:
                #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                #           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

            if True:
                ### training set
                tmp_train_acc, tmp_train_loss = self.run_test(train_loader)
                self.results['train_acc'].append(tmp_train_acc)
                self.results['train_loss'].append(tmp_train_loss)

                ### testing set
                tmp_test_acc, tmp_test_loss = self.run_test(test_loader)
                self.results['test_acc'].append(tmp_test_acc)
                self.results['test_loss'].append(tmp_test_loss)

                print('Epoch [{}/{}]; train_acc={}, train_loss={}; test_acc={}, test_loss={}'.format(epoch + 1,
                                                                                                     EPOCHES,
                                                                                                     tmp_train_acc,
                                                                                                     tmp_train_loss,
                                                                                                     tmp_test_acc,
                                                                                                     tmp_test_loss))
        # Save the model checkpoint
        # torch.save(model.state_dict(), 'model.ckpt')

    def run_test(self, test_loader):
        # Test the model

        self.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        r"""model.eval(): Sets the module in evaluation mode.

               This has any effect only on certain modules. See documentations of
               particular modules for details of their behaviors in training/evaluation
               mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
               etc.
               """

        with torch.no_grad():
            correct = 0.0
            loss = 0.0
            total = 0
            cm = []
            for step, (b_x, b_y) in enumerate(test_loader):
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                b_x = b_x[:, :first_n_pkts * num_features]
                b_x = b_x.view([b_x.shape[0], 1, -1, 1])  # (nSamples, nChannels, x_Height, x_Width)
                b_x = Variable(b_x).float()
                b_y = Variable(b_y).type(torch.LongTensor)
                # b_y_preds = model(b_x)
                b_y_preds, _, _ = self.forward(b_x)
                loss += self.criterion(b_y_preds, b_y)
                _, predicted = torch.max(b_y_preds.data, 1)
                total += b_y.size(0)
                correct += (predicted == b_y).sum().item()

                if step == 0:
                    cm = confusion_matrix(b_y, predicted, labels=[i for i in range(num_classes)])
                    sk_accuracy = accuracy_score(b_y, predicted) * len(b_y)
                else:
                    cm += confusion_matrix(b_y, predicted, labels=[i for i in range(num_classes)])
                    sk_accuracy += accuracy_score(b_y, predicted) * len(b_y)

            # print(cm, sk_accuracy / total)
            # # print('Evaluation Accuracy of the model on the {} samples: {} %'.format(total, 100 * correct / total))

        acc = correct / total
        return acc, loss.data.tolist()


def show_results(data_dict, i=1):
    """

    :param data_dict:
    :param i: first_n_pkts
    :return:
    """
    import matplotlib.pyplot as plt
    # plt.subplots(1,2)

    plt.subplot(1, 2, 1)
    length = len(data_dict['train_acc'])
    plt.plot(range(length), data_dict['train_acc'], 'g-', label='train_acc')
    plt.plot(range(length), data_dict['test_acc'], 'r-', label='test_acc')
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of the first %d pkts' % i)

    plt.subplot(1, 2, 2)
    length = len(data_dict['train_loss'])
    plt.plot(range(length), data_dict['train_loss'], 'g-', label='train_loss')
    plt.plot(range(length), data_dict['test_loss'], 'r-', label='test_loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss of the first %d pkts' % i)

    plt.show()


def get_loader_iterators_contents(train_loader):
    X = []
    y = []
    for step, (b_x, b_y) in enumerate(train_loader):
        X.extend(b_x.data.tolist())
        y.extend(b_y.data.tolist())

    return X, y


def run_main(input_file, n=784):
    # input_file = '../data/data_split_train_v2_711/train_%dpkt_images_merged.csv' % i
    print(input_file)
    dataset = TrafficDataset(input_file, transform=None, normalization_flg=True)

    train_sampler, test_sampler = split_train_test(dataset, split_percent=0.9, shuffle=True)
    cntr = Counter(dataset.y)
    print('dataset: ', len(dataset), ' y:', sorted(cntr.items()))
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=4)  # use all dataset
    train_loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=4, sampler=train_sampler)
    X, y = get_loader_iterators_contents(train_loader)
    cntr = Counter(y)
    print('train_loader: ', len(train_loader.sampler), ' y:', sorted(cntr.items()))
    global test_loader
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=len(test_sampler), num_workers=4,
                                              sampler=test_sampler)
    X, y = get_loader_iterators_contents(test_loader)
    cntr = Counter(y)
    print('test_loader: ', len(test_loader.sampler), ' y:', sorted(cntr.items()))

    model = ConvNet(num_classes, num_features=n).to(device)
    model.run_train(train_loader)
    show_results(model.results, n)

    # model.run_test(test_loader)

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model_%d.ckpt' % n)


def run_main_cross_validation(i):
    input_file = '../data/data_split_train_v2_711/train_%dpkt_images_merged.csv' % i
    print(input_file)
    dataset = TrafficDataset(input_file, transform=None, normalization_flg=True)

    acc_sum = 0.0

    # k_fold = KFold(n_splits=10)
    k_fold = StratifiedKFold(n_splits=10)
    for k, (train_idxs_k, test_idxs_k) in enumerate(k_fold.split(dataset)):
        print('--------------------- k = %d -------------------' % (k + 1))
        cntr = Counter(dataset.y)
        print('dataset: ', len(dataset), ' y:', sorted(cntr.items()))
        train_sampler = SubsetRandomSampler(train_idxs_k)
        test_sampler = SubsetRandomSampler(test_idxs_k)
        # train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=4)  # use all dataset
        train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, num_workers=4,
                                                   sampler=train_sampler)
        X, y = get_loader_iterators_contents(train_loader)
        cntr = Counter(y)
        print('train_loader: ', len(train_idxs_k), ' y:', sorted(cntr.items()))
        global test_loader
        test_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, num_workers=4,
                                                  sampler=test_sampler)
        X, y = get_loader_iterators_contents(test_loader)
        cntr = Counter(y)
        print('test_loader: ', len(test_idxs_k), ' y:', sorted(cntr.items()))

        model = ConvNet(num_classes, num_features=i * 60 + i - 1).to(device)
        print(model)
        model.run_train(train_loader)
        show_results(model.results, i)

        # model.run_test(test_loader)
        acc_sum_tmp = np.sum(model.results['test_acc'])
        if acc_sum < acc_sum_tmp:
            print('***acc_sum:', acc_sum, ' < acc_sum_tmp:', acc_sum_tmp)
            acc_sum = acc_sum_tmp
            # Save the model checkpoint
            torch.save(model.state_dict(), '../results/model_%d.ckpt' % i)


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
    data_path = '../data/fixed-length-transport-layer-payload/session/{}'.format(name_str)
    train_images_file = '{}/{}-byte-payload-per-flow-{}-train-images-idx2-ubyte.gz'.format(data_path, n, name_str)
    train_labels_file = '{}/{}-byte-payload-per-flow-{}-train-labels-idx1-ubyte.gz'.format(data_path, n, name_str)
    test_images_file = '{}/{}-byte-payload-per-flow-{}-test-images-idx2-ubyte.gz'.format(data_path, n, name_str)
    test_labels_file = '{}/{}-byte-payload-per-flow-{}-test-labels-idx1-ubyte.gz'.format(data_path, n, name_str)
    # X_train, X_test = np.expand_dims(idx_reader.read_images(train_images_file), 1), np.expand_dims(
    #     idx_reader.read_images(test_images_file), 1)
    X_train, X_test = idx_reader.read_images(train_images_file,feature_type='payload-len',only_images=True), idx_reader.read_images(test_images_file,feature_type='payload-len',only_images=True)
    y_train, y_test = idx_reader.read_labels(train_labels_file), idx_reader.read_labels(test_labels_file)

    # return X_train, y_train, X_test, y_test
    train_output_file = '../results/%s_%dBytes_train.csv' % (name_str, n)
    with open(train_output_file, 'w') as fid_out:
        (m, n) = X_train.shape
        for row in range(m):
            line = ''
            for col in range(n):
                line += str(X_train[row][col]) + ','
            line += str(int(y_train[row])) + '\n'
            fid_out.write(line)

    test_output_file = '../results/%s_%dBytes_test.csv' % (name_str, n)
    with open(test_output_file, 'w') as fid_out:
        (m, n) = X_test.shape
        for row in range(m):
            line = ''
            for col in range(n):
                line += str(X_test[row][col]) + ','
            line += str(int(y_test[row])) + '\n'
            fid_out.write(line)

    return train_output_file, test_output_file

    # # get data
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

def merge_files(train_output_file1,train_output_file2,train_output_file3):
    import os
    output_file = os.path.splitext(train_output_file1)[0] + '_merged_files.csv'
    with open(output_file, 'w') as fid_out:
        with open(train_output_file1, 'r') as fid_in1:
            line = fid_in1.readline().strip()
            while line:
                line_arr = line.split(',')
                if line_arr[-1] =='2':
                    line_tmp = ','.join(line_arr[:-1]) + ',0\n'
                    fid_out.write(line_tmp)
                line = fid_in1.readline().strip()

        with open(train_output_file2, 'r') as fid_in2:
            line = fid_in2.readline().strip()
            while line:
                line_arr = line.split(',')
                if line_arr[-1] == '2':
                    line_tmp = ','.join(line_arr[:-1]) + ',1\n'
                    fid_out.write(line_tmp)
                line = fid_in2.readline().strip()

        with open(train_output_file3, 'r') as fid_in3:
            line = fid_in3.readline().strip()
            while line:
                line_arr = line.split(',')
                if line_arr[-1] == '2':
                    line_tmp = ','.join(line_arr[:-1]) + ',2\n'
                    fid_out.write(line_tmp)
                line = fid_in3.readline().strip()

    return output_file


if __name__ == '__main__':
    torch.manual_seed(1)

    n = 1000
    # input_file = merge_features_labels('../data/3combined/train_images.csv', '../data/3combined/train_labels.csv')
    # name_str = 'skype'
    # train_output_file1, test_output_file1 = read_skype_sample(name_str, n)
    # name_str = 'facebook'
    # train_output_file2, test_output_file2 = read_skype_sample(name_str, n)
    # name_str = 'hangout'
    # train_output_file3, test_output_file3 = read_skype_sample(name_str, n)
    #
    # train_output_file = merge_files(train_output_file1,train_output_file2,train_output_file3)

    name_str ='vpn-app'
    name_str ='hangout'
    # name_str='skype'
    name_str = 'non-vpn-app'
    train_output_file, test_output_file = read_skype_sample(name_str, n)
    input_file = train_output_file

    remove_labels_lst = [0,6]
    input_file, num_c = remove_special_labels(input_file, remove_labels_lst)
    print(input_file)

    global batch_size, EPOCHES, num_classes, num_features, first_n_pkts
    first_n_pkts = 1
    batch_size = 100
    EPOCHES = 100
    num_classes = num_c
    num_features = 80
    learning_rate = 0.001
    run_main(input_file, num_features * first_n_pkts)

#
# if __name__ == '__main__':
#
#     torch.manual_seed(1)
#     # Hyper parameters
#     num_epochs = 100
#     num_classes = 4
#     batch_size = 64
#     learning_rate = 0.001
#
#     cross_validation_flg = True
#
#     # for i in [1, 3, 5, 8, 10]:
#     for i in [10, 8, 5, 3, 1]:
#         if cross_validation_flg:
#             run_main_cross_validation(i)
#         else:
#             run_main(i)
