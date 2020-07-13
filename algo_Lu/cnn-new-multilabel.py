# Author: kun.bj@outlook.com
# License: XXX
import copy
import os
import random
from collections import Counter
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torchsummary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from torch.autograd import Variable
# np.load issue
# save np.load
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

np_load_old = np.load
# # modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

RANDOM_STATE = 42


def reset_random_state(random_state=42):
    """Reset all states with the same seed

    Parameters
    ----------
    random_state

    Returns
    -------

    """
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(random_state)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(random_state)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(random_state)

    # 4. Set `pytorch` pseudo-random generator at a fixed value
    torch.manual_seed(random_state)
    # # torch.backends.cudnn.deterministic = True
    # # torch.backends.cudnn.benchmark = False
    # # torch.backends.cudnn.enabled = False


reset_random_state(RANDOM_STATE)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class DATA:

    def __init__(self, input_file='', dim=100, n_classes=8):

        self.dim = dim
        self.n_classes = n_classes

        print("loading data.....")
        removelist = [3, 5]  # remove y label, what's reason for this operation?

        a = np.load(input_file)[1:]
        trainY = (a[:, 0]).reshape(-1, 1)  # labels
        trainX = (a[:, 1])  # iat + payload_data
        truetrainX = []
        truetrainY = []
        for i in range(0, len(trainX)):
            if i % 10000 == 0:
                print(i)
            Xli = trainX[i][0]
            Yli = trainY[i][0]
            containinfo = False
            # print(Yli)
            # remove certain data
            if Yli in removelist:
                continue
            elif (Yli == 4):
                Yli = 3
            elif (Yli > 5):
                Yli -= 2

            for s in Xli:
                if s != 0:
                    containinfo = True
                    break

            if (containinfo == True):
                truetrainX.append(Xli)
                truetrainY.append(Yli)
        # fix the length
        truetrainX = np.asarray(truetrainX)[:, :self.dim]
        # multilabel
        new_Y = []
        mapping = {'chat': 1, 'streaming': 1, 'email': 1, 'search': 1}
        # # Google Twitter Youtube Outlook Github Facebook Slack Bing
        apps = {0: 'Google',
                1: 'Twitter',
                2: 'Youtube',
                3: 'Outlook',
                4: 'Github',
                5: 'Facebook',
                6: 'Slack',
                7: 'Bing'}
        app_mapping = {
            'Google': [0, 0, 0, 0, mapping['search']],
            'Twitter': [1, mapping['chat'], mapping['streaming'], 0, 0],
            'Youtube': [2, 0, mapping['streaming'], 0, 0],
            'Outlook': [3, 0, 0, mapping['email'], 0],
            'Github': [4, 0, mapping['streaming'], 0, 0],
            'Facebook': [5, mapping['chat'], mapping['streaming'], 0, 0],
            'Slack': [6, mapping['chat'], 0, 0, 0],
            'Bing': [7, 0, 0, 0, mapping['search']]
        }

        for j, y in enumerate(truetrainY):
            new_Y.append(np.asarray(app_mapping[apps[y]]))

        truetrainY = np.asarray(new_Y, dtype=int)

        print("after load....")
        # print(f"truetrainX[0]: {truetrainX[0]},\ncounter(truetrainY): {counter(truetrainY)}")
        print(truetrainX.shape)
        print(truetrainY.shape)

        # # # truncate
        # truetrainX =((( truetrainX[:,:dim])/255)- 0.5)*2
        # truetrainX = (truetrainX[:,:dim])/255
        # # ss = StandardScaler()
        # # truetrainX = ss.fit_transform(truetrainX[:,:dim])
        # # print("truetrainX1",truetrainX[0][0:20])
        # print("truetrainX2", truetrainX[1][0:20])
        # # print("truetrainX3",truetrainX[2][0:20])
        # # print("truetrainX4",truetrainX[6131][2920:2940])
        # # print("truetrainX5",truetrainX[4230][2920:2940])
        # # print("truetrainX6",truetrainX[520][2920:2940])
        # # resample
        # listdir = {}
        # for i in range(0, len(truetrainY)):
        # 	if truetrainY[i] not in listdir:
        # 		listdir.update({truetrainY[i]: [i]})
        # 	else:
        # 		listdir[truetrainY[i]].append(i)
        # actualdir = {}
        # for i in range(0, n_classes):
        # 	if i in listdir:
        # 		thelist = listdir[i]
        # 	else:
        # 		thelist = []
        # 	if (len(thelist) > SAMPLE):
        # 		actualdir.update({i: random.sample(thelist, SAMPLE)})  # sample 500
        # 	else:
        # 		actualdir.update({i: thelist})
        # listdir = {}
        # dic = {}
        # truetruetrainX = []
        # truetruetrainY = []
        # for i in range(0, len(truetrainY)):
        # 	if i not in actualdir[truetrainY[i]]:
        # 		continue
        # 	truetruetrainX.append(truetrainX[i])
        # 	truetruetrainY.append(truetrainY[i])
        # X = np.asarray(truetruetrainX)
        # Y = np.asarray(truetruetrainY)
        #
        # if resample == False:
        # 	X = truetrainX  # FOR non sample
        # 	Y = truetrainY
        # # for lstm
        # print("X.shape[0]", X.shape[0])
        # print("X.shape[1]", X.shape[1])
        # #
        # # # X = X.reshape(X.shape[0],X.shape[1],1) # necessary for lstm!!

        self.X = truetrainX
        self.y = truetrainY

    def sampling(self, n=100, random_state=42):

        n_each = int(n / self.n_classes)

        # X, y = resample(self.X, self.y, n_samples=n, random_state=random_state, replace=False)
        X, y = self.balanced_subsample(self.X, self.y, n_each_label=n_each, random_state=random_state)
        # print(f'X.shape: {X.shape}, y.shape: {y.shape}, Counter(y): {counter(y)}')
        return X, y

    def balanced_subsample(self, X, y, n_each_label=100, random_state=42):
        """
        :param x:
        :param y:
        :param :
        :return:
        """
        X_new = []
        y_new = []

        for i, _yi in enumerate(np.unique(y[:, 0])):
            Xi = X[(y[:, 0] == _yi)]
            yi = y[(y[:, 0] == _yi)]
            if len(yi) > n_each_label:
                Xi, yi = resample(Xi, yi, n_samples=n_each_label, random_state=random_state, replace=False)
            else:
                Xi, yi = resample(Xi, yi, n_samples=n_each_label, random_state=random_state, replace=True)
            if i == 0:
                X_new = deepcopy(Xi)
                y_new = deepcopy(yi)
            else:
                X_new = np.concatenate([X_new, Xi], axis=0)
                y_new = np.concatenate([y_new, yi], axis=0)

        return X_new, y_new


class Flatten(nn.Module):
    # def __init__(self):
    #     self._out_dim = 1
    def forward(self, input):
        self._out_dim = input.view(input.size(0), -1)[1]
        return input.view(input.size(0), -1)


class Discriminator(nn.Module):
    def __init__(self, img_shape, sl=1, q=0.3, n_ksi=10, center=None, m=2, n_classes=[]):
        """Network architecture

        Parameters
        ----------
        img_shape
        sl
        """
        self.img_shape = img_shape
        self.sl = sl  # scalar
        super(Discriminator, self).__init__()
        self.n_classes = n_classes

        in_dim = int(np.prod(self.img_shape))
        h_dim = int(in_dim * self.sl) or 1
        self.q = q
        self.n_layers = m
        # self.encoder = nn.Sequential(
        #     nn.Linear(in_dim, h_dim),
        #     nn.LeakyReLU(self.q, inplace=True),
        #     # nn.Tanh(),
        #     # nn.Sigmoid(),
        #     nn.Dropout(),
        #     nn.Linear(h_dim, h_dim),
        #     nn.LeakyReLU(self.q, inplace=True),
        #     # nn.Tanh(),
        #     # nn.Sigmoid(),
        #     nn.Dropout(),
        #
        #     nn.Linear(h_dim, h_dim),
        #     nn.LeakyReLU(self.q, inplace=True),
        #     # nn.Tanh(),
        #     # nn.Sigmoid(),
        #     # nn.Dropout(),
        #
        #     # nn.Tanh(),
        #     # nn.Sigmoid(),
        #     # nn.Dropout(),
        #     # nn.Linear(h_dim, 1, bias=False), # for autoencoder
        #     nn.Linear(h_dim, self.n_classes[0] + self.n_classes[1]),
        #     # nn.Sigmoid()
        # )

        stride = 1
        kernel_size = 3
        # create input layer
        self.encoder = nn.Sequential()
        # self.encoder.add_module('linear_in', nn.Linear(in_dim, h_dim))
        out_channels = 128
        input_layer = torch.nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        _out_dim = (in_dim - (kernel_size - stride)) // stride
        self.encoder.add_module(f'input_layer', input_layer)
        # self.encoder.add_module('leakyrelu_in', nn.LeakyReLU(self.q))
        self.encoder.add_module('elu_in', nn.ELU())
        self.encoder.add_module('dropout_in', nn.Dropout())

        # create hidden layers
        self.n_layers = 4
        in_channels = out_channels  # each channel has number of filters
        # out_channels = int(in_channels //2)
        out_channels = 32
        for i in range(self.n_layers):
            # self.encoder.add_module(f'linear_{i + 1}', nn.Linear(h_dim, h_dim))
            kernel_size = i + 1
            layer_i = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
            _out_dim = (_out_dim - (kernel_size - stride)) // stride
            self.encoder.add_module(f'linear_{i + 1}', layer_i)
            # self.encoder.add_module(f'leakyrelu_{i+1}', nn.LeakyReLU(self.q))
            self.encoder.add_module(f'elu_{i + 1}', nn.ELU())
            # self.encoder.add_module(nn.Tanh()),
            # self.encoder.add_module(nn.Sigmoid()),
            # self.encoder.add_module(nn.Dropout())
            in_channels = int(out_channels)
            # out_channels = int(in_channels // 2)
            # print(out_channels)

        # create ouput layer
        # out = out.view(x.shape[0], out.size(1) * out.size(2))
        self.encoder.add_module('flatten', Flatten())
        # _out_dim = list(self.encoder)[-1]._out_dim
        _out_dim = _out_dim * out_channels
        self.encoder.add_module('linear_full_1', nn.Linear(_out_dim, 100))
        # self.encoder.add_module('linear_full_2', nn.Linear(500, 100))
        self.encoder.add_module('linear_out', nn.Linear(100, self.n_classes[0] + self.n_classes[1]))

        # self.decoder = nn.Sequential(
        #     nn.Linear(1, int(h_dim * self.sl) or 1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Tanh(),
        #     # nn.Sigmoid(),
        #     # nn.Dropout(),
        #     nn.Linear(int(h_dim * sl) or 1, int(h_dim * self.sl) or 1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Tanh(),
        #     # nn.Sigmoid(),
        #     # nn.Dropout(),
        #     nn.Linear(int(h_dim * self.sl) or 1, in_dim, bias=False),   # the bias of the last layer of decoder must be False
        #     nn.Tanh()
        #     # nn.LeakyReLU(0.2, inplace=True)
        # )

    def forward(self, img):
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width
        # img_flat = img.view(img.size(0), -1)
        img_flat = img.view(img.size(0), 1, -1)

        encoded_feats = self.encoder(img_flat)
        # bound = 50  # torch.exp(torch.tensor(5e+1))= tensor(5.1847e+21)
        # encoded_feats = torch.tensor(1.0 / np.sqrt(2 * np.pi)) * torch.exp(
        #     -(encoded_feats ** 2 / 2).clamp(-bound, bound))  # normal distribution pdf with mu=0, std=1
        return encoded_feats, ''


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    y = np.asarray(y, dtype=int)
    return np.eye(num_classes, dtype='uint8')[y]


class NNDetector():

    def __init__(self, n_epoch, bs, lr, alpha, alpha2, sl, m, params={}, n_classes=[], X_train=''):
        self.n_epoch = n_epoch
        self.bs = bs
        self.lr = lr
        self.alpha = alpha
        self.alpha2 = alpha2
        self.sl = sl  # scalar: hid_dim = int(scalar * feat_dim)
        self.m = m
        self.random_state = params['random_state']
        self.out_dir = os.path.join(params['out_dir'], params['data_file'])
        self.sub_dir = f'NNDetector-n_epoch:{self.n_epoch}-bs:{self.bs}-lr:{self.lr}-a:{self.alpha}-a2:{self.alpha2}-sl:{self.sl}-m:{self.m}'
        self.n_classes = n_classes

        self.gs = params['gs']
        self.show = params['show']

        reset_random_state(random_state=self.random_state)

        # latent_dim = X_train.shape[1]
        # n_ksi = int(X_train.shape[0]* self.sl)
        self.N = int(X_train.shape[0])
        # Initialize discriminator
        self.center = np.mean(X_train, axis=0)

        # self.n_classes = [np.unique(y_train[:,0]), np.unique(y_train[:,1])]
        print(f'self.n_classes: {self.n_classes}')
        self.discriminator = Discriminator(img_shape=(X_train.shape[1],), sl=self.sl, q=self.alpha, n_ksi=self.N,
                                           center=self.center, m=self.m, n_classes=self.n_classes)
        # self.C = 30  # hyperparameters
        # discriminator = Discriminator(img_shape=(self.C,), sl=self.sl)
        # adversarial_loss = torch.nn.BCELoss()
        adversarial_loss = torch.nn.MSELoss()
        print(f'{torchsummary.summary(self.discriminator, input_size=(X_train.shape[1],))}')
        # if cuda:
        #     discriminator.cuda()
        #     adversarial_loss.cuda()

        # Optimizers
        b1 = 0.5
        b2 = 0.999
        # self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(b1, b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(b1, b2),
                                            weight_decay=self.lr / 10)

        # adaptive learning rate
        self.scheduler = ReduceLROnPlateau(self.optimizer_D, 'min', patience=10)

    def train(self, X_train, y_train, X_test, y_test):
        # ----------
        #  Training
        # ----------
        tensor_x = torch.Tensor(X_train)  # transform to torch tensor
        tensor_y = torch.Tensor(y_train)

        my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset

        dataloader = torch.utils.data.DataLoader(
            my_dataset,
            batch_size=X_train.shape[0] if self.bs == -1 else self.bs,
            shuffle=True
        )

        results = {'g_losses': [], 'd_losses': [], 'params': {'n_epoch': self.n_epoch, 'bs': self.bs,
                                                              'lr': self.lr, 'a': self.alpha, 'a2': self.alpha2,
                                                              'sl': self.sl},
                   'mse_losses': [], 'mean_losses': [],
                   'std_losses': [],
                   'mean_outs': [], 'std_outs': [],
                   'imgs': []
                   }
        debug = 1
        self.history = results
        self.discriminator.train(mode=True)
        prev_params = copy.deepcopy([v.data.numpy().tolist() for v in list(self.discriminator.parameters())])
        ksi = torch.empty(X_train.shape[1], requires_grad=False).normal_(mean=0, std=1) ** 2
        for epoch in range(self.n_epoch):
            g_losses = []
            d_losses = []
            g_fake_losses = []
            mse_losses = []
            mean_losses = []
            std_losses = []
            mean_outs = []
            std_outs = []

            if self.show:
                curr_params = [v.data.numpy().tolist() for v in list(self.discriminator.parameters())]
                # print('curr_params: ', curr_params)
                # print("diff: ", diff_params(prev_params, curr_params))
            for i, (imgs, labels) in enumerate(dataloader):
                # if i == 0: print(imgs)
                # Configure input
                real_imgs = Variable(imgs.type(Tensor))
                # prev_mean = torch.mean(real_imgs)
                self.optimizer_D.zero_grad()
                out_encoder, _ = self.discriminator(real_imgs)

                y1 = to_categorical(labels[:, 0].data.numpy(), num_classes=self.n_classes[0])
                # y2= to_categorical(labels[:, 1].data.numpy(), num_classes=self.n_classes[1])
                y2 = labels[:, 1:]

                y1 = Variable(torch.Tensor(y1).type(Tensor))
                y2 = Variable(torch.Tensor(y2).type(Tensor))
                # mean_out = torch.mean(out, dim=0, keepdim=True)  # if batch_size =1, then out == mean_out, d_loss=0.0
                # std_out = torch.std(out, dim=0, keepdim=True)
                # # mse: (input-target).pow(2).sum()/ input.numel()   # input.numel() !=  batch_size
                # # loss = mse + dist(mean, 0) + dist(std, 1), which requires min(mse) and also keeps mean = 0, std=1
                # unit_std = Variable(Tensor(1, imgs.size(1)).fill_(1.0), requires_grad=False)
                # mse_loss = adversarial_loss(out, mean_out.expand(*imgs.shape)) * X_train.shape[1]
                # mean_loss = torch.norm(mean_out, p=2) ** 2
                # std_loss = torch.norm(std_out - unit_std, p=2) ** 2
                # d_loss = mse_loss
                # # d_loss = mse_loss + mean_loss + std_loss
                # d_loss = mse_loss + self.alpha * mean_loss
                # # d_loss = mse_loss + self.alpha * (1 / mean_loss)
                # # d_loss = mse_loss + self.alpha * (mean_loss + std_loss)
                # # d_loss = mse_loss + self.alpha * (mean_loss) + self.alpha2 * std_loss
                # # d_loss = d_loss * 100
                class1_loss = torch.sum((nn.functional.softmax(out_encoder[:, :8], dim=1) - y1) ** 2)
                class2_loss = torch.sum((torch.sigmoid(out_encoder[:, 8:]) - y2) ** 2)
                # out = torch.sum((out_encoder-labels)**2, dim=1)
                out = class1_loss + self.alpha * class2_loss
                # # out = torch.exp(-torch.sum((out - real_imgs) ** 2, dim=1) / imgs.shape[1])
                d_loss = torch.sum(out) / imgs.shape[
                    0]  # + self.alpha* np.mean(elipson) # R**2 + alpha* np.mean(elipson)

                d_losses.append(d_loss.item())

                d_loss.backward()

                self.optimizer_D.step()

                # mse_losses.append(mse_loss.item())
                # mean_losses.append(mean_loss.item())
                # std_losses.append(std_loss.item())
                # mean_outs.append(mean_out.data.numpy().ravel().reshape(1, -1))
                # std_outs.append(std_out.data.numpy().ravel().reshape(1, -1))

            results['d_losses'].append(np.mean(d_losses))  # scalar
            # results['mse_losses'].append(np.mean(mse_losses))  # scalar
            # results['mean_losses'].append(np.mean(mean_losses))  # scalar
            # results['std_losses'].append(np.mean(std_losses))  # scalar
            # results['mean_outs'].append(np.mean(mean_outs, axis=0))  # vectors
            # results['std_outs'].append(np.mean(std_outs, axis=0))  # vectors

            # adjust learning rate
            # scheduler.step(d_loss)
            # print(optimizer_D.param_groups[0]['lr'])
            if self.show:
                prev_params = curr_params

            # https://discuss.pytorch.org/t/get-current-lr-of-optimizer-with-adaptive-lr/24851/5
            curr_lr = self.optimizer_D.param_groups[0]['lr']
            # print(f'epoch:{epoch}, {np.mean(d_losses)}, {np.mean(mse_losses)}, {np.mean(mean_losses)}, '
            #       f'{np.mean(std_losses)},{np.mean(mean_outs, axis=0)}, {np.mean(std_outs, axis=0)}, {curr_lr}, {self.sub_dir}')
            # print(f'epoch:{epoch}, {np.mean(d_losses)}, {curr_lr}, {self.sub_dir}, {[(name, params) for name, params in discriminator.named_parameters()]}')
            # print(discriminator.encoder[0].weight.data.numpy() == discriminator.decoder[4].weight.data.numpy().T)
            print(f'epoch:{epoch}, {np.mean(d_losses)}, {curr_lr}, {self.sub_dir}')

            # get thres
            self.test(X_test, y_test, name=f'val-epoch_{epoch}')

            # opencv doesn't support .pdf, so here we use .png
            out_file = f'{self.out_dir}/{self.sub_dir}/train-epoch:{epoch}.png'
            # plot_blobs(X_train_out, y_train, out_file,
            #            title=out_file, random_state=self.random_state, show=self.show,
            #            xlim=(-2, 2), ylim=(-2, 2))
            results['imgs'].append(out_file)

            self.discriminator.train()  # Sets the module in training mode.

        results['g_losses'] = results['mean_outs']

        # imgs to animations
        out_file = f'{self.out_dir}/{self.sub_dir}/train_epochs:{epoch + 1}.mp4'
        # imgs2video(results['imgs'], out_file)

        # get thres
        self.discriminator.eval()
        X_train_out, _ = self.discriminator(Tensor(X_train))
        # X_train_out = X_train_out.data.numpy().ravel()
        X_train_out = X_train_out.data.numpy()
        out_file = f'{self.out_dir}/{self.sub_dir}/train_out.pdf'

        mean_thres = np.mean(X_train_out, axis=0)

        q = 0.9
        self.R_squared = np.quantile(X_train_out, q=q)

        print(f'q: {q}, R_squared: {self.R_squared}, quatiles: {np.quantile(X_train_out, q=[0.1, 0.3, 0.5, 0.7, 0.9])}')

        # results['mean_thres'] = mean_thres  # scaler
        # y_score = np.linalg.norm(X_train_out - mean_thres, ord=2, axis=1) ** 2
        # y_score = [np.sum((v-mean_thres)**2) for v in X_train_out]
        # print(y_score)
        # # # the interquartile range (IQR)= Q3-Q1 (75th - 25th)
        # vs = np.quantile(X_train_out, [0.1, 0.5,  0.9])
        # IQR_thres = (vs[-1] - vs[0]) /2 + vs[0]
        # print(f'mean_thres: {mean_thres}, IQR_thres: {IQR_thres}, median_thres: {vs[1]}')
        # results['IQR_thres'] = IQR_thres
        # results['X_train_out'] = X_train_out

        # self.discriminator = discriminator

        self.thres = mean_thres
        self.history['thres'] = mean_thres
        self.history['R_squred'] = self.R_squared

        return self

    def test(self, X_test, y_test, name='test'):
        # print('thres:', self.thres)
        # y_score = discriminator(X_test)
        X_test = Tensor(X_test)
        # valid = Variable(Tensor(len(y_test), 1).fill_(1.0), requires_grad=False)

        self.discriminator.eval()
        X_test_out, _ = self.discriminator(X_test)

        # # X_test_out = X_test_out.data.numpy().ravel()
        # X_test_out = X_test_out.data.numpy()
        # q = [0.1, 0.3, 0.5, 0.7, 0.9]

        # for all labels
        correct_num = 0
        y_pred_labels = []

        for i, (y1, y2) in enumerate(
                zip(nn.functional.softmax(X_test_out[:, :8], dim=1), torch.sigmoid(X_test_out[:, 8:]))):
            y1 = y1.data.numpy()
            y2 = y2.data.numpy()
            # y1 = np.argmax(y1, axis=1)
            y1 = np.argmax(y1)
            correct_flg = 0
            _lab = [y1]
            for j in range(self.n_classes[1]):
                if y2[j] > 0.5:
                    y2[j] = 1
                else:
                    y2[j] = 0

                if y2[j] == y_test[i][j + 1]:
                    correct_flg += 1

                _lab.append(y2[j])
            y_pred_labels.append(np.asarray(_lab, dtype=int))
            if y1 == y_test[i, 0] and correct_flg == self.n_classes[1]:
                correct_num += 1

        y_pred_labels = np.asarray(y_pred_labels, dtype=int)

        # totals
        acc = correct_num / len(y_test)
        auc = acc

        accs = []
        # each class accuracy
        for i in range(y_pred_labels.shape[1]):
            acc_i = [1 for v1, v2 in zip(y_pred_labels[:, i], y_test[:, i]) if v1 == v2]
            acc_i = len(acc_i) / y_pred_labels.shape[0]
            accs.append(acc_i)

        # print(f'{name} global acc: {acc}, and each class\'s acc {accs}')
        print(f'{name} global acc: {acc}, and each class\'s acc {accs}')
        self.history['X_test_out'] = X_test_out
        self.history['X_test'] = X_test
        self.history['y_test'] = y_test
        self.auc = auc
        self.history['auc'] = auc

        # if i == 0:
        #     y_score = y_score.data.numpy()
        #     print(f'{name} y_score: {np.quantile(y_score, q=[0.1, 0.3, 0.5, 0.7, 0.9])}')
        #     # fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
        #     # auc = metrics.auc(fpr, tpr)
        #     y_pred = [np.argmax(v) for v in y_score]
        #     cm = metrics.confusion_matrix(y_test[:, i], y_pred)
        #     auc = metrics.accuracy_score(y_test[:, i], y_pred)
        #     print(f'{name} cm of class_{i}: {cm}, acc: {auc}')
        #
        #     self.history['X_test_out'] = y_score
        #     self.history['X_test'] = X_test
        #     self.history['y_test'] = y_test
        #     self.auc = auc
        #     self.history['auc'] = auc
        #     if i == 0:
        #         self.accuracy = auc
        # else:
        #     y_score = y_score.data.numpy()
        #     for j in range(4):
        #         _y_score = y_score[:, j]
        #         print(f'{j} {name} y_score: {np.quantile(_y_score, q=[0.1, 0.3, 0.5, 0.7, 0.9])}')
        #         # fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
        #         # auc = metrics.auc(fpr, tpr)
        #         y_pred = [np.argmax(v) for v in _y_score]
        #         cm = metrics.confusion_matrix(y_test[:, i], y_pred)
        #         auc = metrics.accuracy_score(y_test[:, i], y_pred)
        #         print(f'{j} {name} cm of class_{i}: {cm}, acc: {auc}')
        #
        #         # self.history['X_test_out'] = y_score
        #         # self.history['X_test'] = X_test
        #         # self.history['y_test'] = y_test
        #         # self.auc = auc
        #         # self.history['auc'] = auc
        #         # if i == 0:
        #         #     self.accuracy = auc

        self.F1 = 0
        self.accuracy = acc
        self.recall = 0
        self.precision = 0
        # self.accuracy = auc
        self.train_time = 0
        self.test_time = 0
        # # mean_thres
        # mean_thres = self.history['mean_thres']
        # print('mean_thres:', mean_thres)
        # y_score_mean_thres = [abs(v[0] - mean_thres) for v in
        #            d_out.tolist()]  # normal value has a small score, abnormal value has a large score
        # fpr_mean_thres, tpr_mean_thres, _ = roc_curve(y_test, y_score_mean_thres, pos_label=1)
        # auc_mean_thres = metrics.auc(fpr_mean_thres, tpr_mean_thres)
        #
        # self.history['X_test_out_mean_thres'] = y_score_mean_thres
        # self.history['auc_mean_thres'] = auc_mean_thres

        return auc


class VISUAL:

    def __init__(self):
        pass

    def show(self):
        # print("score", score)
        # with open(pyname,"w") as wfile:
        # 	wfile.write("import matplotlib.pyplot as plt\n")
        # 	wfile.write("acc = {0}\n".format(history.acc))
        # 	wfile.write("valacc = {0}\n".format(history.valacc))
        # 	wfile.write("plt.plot(acc)\nplt.plot(valacc)\nplt.title('Model accuracy')\nplt.ylabel('Accuracy')\nplt.xlabel('Epoch')\nplt.legend(['Train', 'Test'], loc='upper left')\nplt.savefig('"+modelname+str(dim)+ts+".jpg')")
        #
        # print(metrics.confusion_matrix(y_true=y_test1, y_pred=y_test_pred))
        # print(metrics.confusion_matrix(y_true=y_train1, y_pred=y_train_pred))
        # plot_model(model, to_file='model'+modelname+str(dim)+ts+'.png')
        pass

    def show_online(self, results):
        import matplotlib.pyplot as plt
        # from matplotlib import rcParams
        # rcParams.update({'figure.autolayout': True})

        fig, ax = plt.subplots()
        accuracys = [v['accuracy'] for k, v in results.items()]
        plt.plot(range(len(accuracys)), accuracys)
        F1 = [v['F1'] for k, v in results.items()]
        plt.plot(range(len(F1)), F1)
        recalls = [v['recall'] for k, v in results.items()]
        plt.plot(range(len(recalls)), recalls)
        precisions = [v['precision'] for k, v in results.items()]
        plt.plot(range(len(precisions)), precisions)

        plt.xlabel('Online test after each updated model')
        plt.ylabel('accuracy, F1, recall, and precision')
        # # plt.title('F1 Scores by category')
        # plt.xticks(index + 2.5 * bar_width,
        #            (app[i], app[i + 1], app[i + 2], app[i + 3], app[i + 4], app[i + 5], app[i + 6], app[i + 7]))
        plt.tight_layout()

        plt.legend(['accuracy', 'F1', 'recall', 'Precision'], loc='lower right')
        # plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
        plt.savefig("F1_for_all.pdf")  # should use before plt.show()
        plt.show()


def counter(y_test):
    res = dict(sorted(Counter(y_test).items()))

    return res


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1 / 9)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_val = ss.fit_transform(X_val)
    X_test = ss.fit_transform(X_test)

    print(f'X.shape: {X.shape}, y.shape: {y.shape}')
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def main():
    results = {}  # store all the results
    dim = 3000  # feature dimensions
    # Google Twitter Youtube Outlook Github Facebook Slack Bing
    N_CLASSES = 8  # number of applications

    ########################################################################
    # offline train and test
    input_file = 'data/app_clf/trdata_PT_8000.npy'
    dt = DATA(input_file, dim=dim, n_classes=N_CLASSES)  # feature length. It must be larger than 15.
    X, y = dt.sampling(n=700 * N_CLASSES, random_state=RANDOM_STATE)

    detectors = ['NNDetector']  # ['OCSVM', 'NNDetector']
    params = {'show': False, 'gs': False, 'random_state': RANDOM_STATE, 'out_dir': '', 'overwrite': False,
              'n_jobs': 1, 'data_file': input_file}

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)

    # model = ODCNN(in_dim=X.shape[1], epochs=20, batch_size=64, learning_rate=1e-5, n_classes=N_CLASSES)
    model = NNDetector(n_epoch=100, bs=64, lr=1e-4, alpha=0.1, alpha2=0.1, sl=1, m=0, params=params,
                       n_classes=[N_CLASSES, y_train.shape[1] - 1], X_train=X_train)

    model.train(X_train, y_train, X_val, y_val)
    # evaluate on train set
    model.test(X_train, y_train, name='train')
    # evaluate on test set
    model.test(X_test, y_test)

    results[0] = {'accuracy': model.accuracy, 'F1': model.F1, 'recall': model.recall, 'precision': model.precision,
                  'train_time': model.train_time, 'test_time': model.test_time}
    ########################################################################
    # Retrain and online test
    retrain_online_test = 1
    cnt = 1
    while retrain_online_test and cnt <= 10:
        # new_data
        print(f'\n\n=== Retrain and Online test {cnt}...')
        X, y = dt.sampling(n=700 * N_CLASSES, random_state=cnt * RANDOM_STATE)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)

        model.train(X_train, y_train, X_val, y_val)
        # evaluate on train set
        model.test(X_train, y_train, name='train')
        # evaluate on test set
        model.test(X_test, y_test)
        results[cnt] = {'accuracy': model.accuracy, 'F1': model.F1, 'recall': model.recall,
                        'precision': model.precision, 'train_time': model.train_time, 'test_time': model.test_time}
        cnt += 1

    # show the results
    pprint(results)
    vs = VISUAL()
    vs.show_online(results)


if __name__ == '__main__':
    main()
