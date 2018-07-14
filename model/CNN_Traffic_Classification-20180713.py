# -*- coding:utf-8 -*-
"""

    refer: https://raw.githubusercontent.com/yunjey/pytorch-tutorial/master/tutorials/02-intermediate/convolutional_neural_network/main.py
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

# Device configuration
from preprocess.TrafficDataset import TrafficDataset, split_train_test

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10, num_features=60):

        # self.train_loader= train_loader
        super(ConvNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x1 square convolution
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 1), stride=1),
            # nn.BatchNorm2d(16),
            nn.Tanh(),
            # nn.MaxPool2d(kernel_size=(2,1), stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 1), stride=1, padding=0),
            # nn.BatchNorm2d(32),
            nn.Tanh(),
            # nn.MaxPool2d(kernel_size=(2,1), stride=2)
        )
        self.fc = nn.Linear(32 * (num_features - (5 - 1) - (3 - 1)) * 1,
                            num_classes)  # (1, 16, 60*i +i-1-(5-1),1) -> (16, 32, 60*i +i-1-(5-1) -(3-1),1)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        # out= nn.Softmax(out)
        return out

    def run_train(self, train_loader, mode=True):
        # Train the model
        self.results = {}
        self.results['train_acc'] = []
        self.results['train_loss'] = []

        self.results['test_acc'] = []
        self.results['test_loss'] = []

        total_step = len(train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                images = images.view([images.shape[0], 1, -1, 1])
                images = Variable(images).float()
                labels = Variable(labels).type(torch.LongTensor)
                # Forward pass
                # outputs = model(images)
                outputs = self.forward(images)
                loss = self.criterion(outputs, labels)

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
                                                                                                     num_epochs,
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
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                images = images.view([images.shape[0], 1, -1, 1])  # (nSamples, nChannels, x_Height, x_Width)
                images = Variable(images).float()
                labels = Variable(labels).type(torch.LongTensor)
                # outputs = model(images)
                outputs = self.forward(images)
                loss += self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # print('Evaluation Accuracy of the model on the {} images: {} %'.format(total, 100 * correct / total))

        acc = correct / total
        return acc, loss


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


def run_main(i):
    input_file = '../data/data_split_train_v2_711/train_%dpkt_images_merged.csv' % i
    print(input_file)
    dataset = TrafficDataset(input_file, transform=None, normalization_flg=True)

    train_sampler, test_sampler = split_train_test(dataset, split_percent=0.7, shuffle=True)
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=30, shuffle=True, num_workers=4, sampler=train_sampler)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=4, sampler=train_sampler)
    global test_loader
    test_loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=4, sampler=test_sampler)

    model = ConvNet(num_classes, num_features=i * 60 + i - 1).to(device)
    model.run_train(train_loader)
    show_results(model.results, i)

    # model.run_test(test_loader)

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model_%d.ckpt' % i)


if __name__ == '__main__':

    torch.manual_seed(1)
    # Hyper parameters
    num_epochs = 200
    num_classes = 4
    batch_size = 32
    learning_rate = 0.001

    for i in [1, 3, 5, 8, 10]:
        run_main(i)
