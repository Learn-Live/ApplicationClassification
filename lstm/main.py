import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import utils
from data import TextLoader
from model import LSTMClassifier

import time
from collections import Counter
from copy import deepcopy
from pprint import pprint

import keras.callbacks
import numpy as np
import sklearn.metrics as metrics
from keras import optimizers
from keras.layers import Conv1D, Reshape, Dense, Dropout, \
    Flatten
from keras.models import Sequential
from keras.regularizers import l1_l2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# np.load issue
# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

RANDOM_STATE = 42


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
        truetrainY = np.asarray(truetrainY)
        print("after load....")
        print(f"truetrainX[0]: {truetrainX[0]},\ncounter(truetrainY): {counter(truetrainY)}")
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
        print(f'X.shape: {X.shape}, y.shape: {y.shape}, Counter(y): {counter(y)}')
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

        for i, _yi in enumerate(np.unique(y)):
            Xi = X[(y == _yi)]
            yi = y[(y == _yi)]
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


class AccuracyHistory(keras.callbacks.Callback):
    def __init__(self):
        pass

    def on_train_begin(self, logs={}):
        self.acc = []
        self.valacc = []
        self.loss = []
        self.valloss = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.valacc.append(logs.get('val_acc'))
        self.loss.append(logs.get('loss'))
        self.valloss.append(logs.get('val_loss'))
        print('acc = ', self.acc)
        print('val_acc = ', self.valacc)
        print('loss = ', self.loss)
        print('val_loss = ', self.valloss)


class ODCNN:
    def __init__(self, in_dim=100, epochs=2, batch_size=32, learning_rate=1e-5, n_classes=8, random_state=42):
        self.in_dim = in_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_classes = n_classes

        def construct_model(dim):
            '''
            #model lstm
            modelname = "LSTM"
            model = Sequential()
            #model.add(LSTM(80))
            model.add(Bidirectional(LSTM(100,activation = 'relu', input_shape=(dim, 1))))
            #model.add(Dense(50,activation= 'relu'))
            model.add(Dense(n_classes ,activation = 'softmax'))
            '''

            modelname = "CNN"
            model = Sequential()
            model.add(Reshape((dim, 1), input_shape=(dim,)))
            model.add(Conv1D(16, strides=1, kernel_size=3, activation="relu"))  # 32
            # model.add(LeakyReLU(alpha=0.1))
            # model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(16, strides=1, kernel_size=3, activation="relu"))  # 16
            # model.add(LeakyReLU(alpha=0.1))
            # model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.2))

            model.add(Conv1D(32, kernel_size=3, activation='relu'))
            model.add(Conv1D(32, kernel_size=3, activation='relu'))
            # model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.2))
            # model.add(GlobalAveragePooling1D())
            # model.add(Conv1D(8,strides =1, kernel_size = 3)) #16
            # model.add(LeakyReLU(alpha=0.1))
            # model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())  # this layer ok?
            model.add(Dense(500, activation='relu'))
            # model.add(LeakyReLU())
            model.add(Dense(300, activation='relu'))
            # model.add(LeakyReLU())
            # model.add(Dense(20, activation = 'relu'))
            # model.add(LeakyReLU())
            model.add(Dense(self.n_classes, activation='softmax', activity_regularizer=l1_l2()))
            # model.add(Dense(8, activation='softmax'))
            '''
            #model MLP
            model = Sequential()
            model.add(Dense(20, input_dim=dim, activation='relu'))
            #model.add(Dense(200, activation='relu'))
            model.add(Dense(n_classes , activation='softmax'))
            '''
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=optimizers.Adam(lr=self.learning_rate),
                          metrics=['accuracy'])
            print(model.summary())

            return model

        self.model = construct_model(self.in_dim)

        self.history = AccuracyHistory()

    def train(self, X_train, y_train, X_val, y_val):
        timestart = time.time()

        y_train = to_categorical(y_train, num_classes=self.n_classes)
        y_val = to_categorical(y_val, num_classes=self.n_classes)

        self.model.fit(X_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=1,
                       validation_data=(X_val, y_val),
                       callbacks=[self.history])
        # print(model.summary())

        timeend = time.time()
        self.train_time = timeend - timestart
        print("Training time: ", self.train_time)

    def test(self, X_test, y_test, name='test'):
        # statistic parts:
        timestart = time.time()

        y_test_pred = self.model.predict(X_test)
        y_test_pred = np.argmax(y_test_pred, axis=1)

        timeend = time.time()
        self.test_time = timeend - timestart
        print(f"Testing time on {name} set: ", self.test_time)

        print(f'{name} set, Counter(y): {counter(y_test)}')
        print(f'cm of {name} set: ', metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred))
        report = metrics.classification_report(y_test, y_test_pred)
        # print(report)
        self.accuracy = metrics.accuracy_score(y_test, y_test_pred)
        self.F1 = metrics.f1_score(y_test, y_test_pred, average='weighted')
        self.recall = metrics.recall_score(y_test, y_test_pred, average='weighted')
        self.precision = metrics.precision_score(y_test, y_test_pred, average='weighted')
        print(f'{name} set, accuracy: {self.accuracy}, F1: {self.F1}, recall: {self.recall}, '
              f'precision: {self.precision}')
        # y_train_pred = self.model.predict(X_train)
        # y_train_pred = np.argmax(y_train_pred , axis=1)
        # print(f'cm of train set: ', metrics.confusion_matrix(y_true=y_train, y_pred=y_train_pred))

        # pyname = "model_train_history"+modelname+str(dim)+ts+".py"
        # # #print("ts:",ts)
        #
        # score = self.model.evaluate(X_test, y_test, verbose=0)
        # print("score", score)
        # with open(pyname, "w") as wfile:
        # 	wfile.write("import matplotlib.pyplot as plt\n")
        # 	wfile.write("acc = {0}\n".format(history.acc))
        # 	wfile.write("valacc = {0}\n".format(history.valacc))
        # 	wfile.write(
        # 		"plt.plot(acc)\nplt.plot(valacc)\nplt.title('Model accuracy')\nplt.ylabel('Accuracy')\nplt.xlabel('Epoch')\nplt.legend(['Train', 'Test'], loc='upper left')\nplt.savefig('" + modelname + str(
        # 			dim) + ts + ".jpg')")
        #
        # print(metrics.confusion_matrix(y_true=y_test1, y_pred=y_test_pred))
        # print(metrics.confusion_matrix(y_true=y_train1, y_pred=y_train_pred))
        # plot_model(model, to_file='model' + modelname + str(dim) + ts + '.png')


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

    # ss = StandardScaler()
    # X_train = ss.fit_transform(X_train)
    # X_val = ss.fit_transform(X_val)
    # X_test = ss.fit_transform(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def apply(model, criterion, batch, targets, lengths):
    pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
    loss = criterion(pred, torch.autograd.Variable(targets))
    return pred, loss


def train_model(model, optimizer, train, dev, x_to_ix, y_to_ix, batch_size, max_epochs):
    criterion = nn.NLLLoss(size_average=False)
    for epoch in range(max_epochs):
        print('Epoch:', epoch)
        y_true = list()
        y_pred = list()
        total_loss = 0
        for batch, targets, lengths, _ in utils.create_dataset(train, x_to_ix, y_to_ix, batch_size=batch_size):
            batch, targets, lengths = utils.sort_batch(batch, targets, lengths)
            model.zero_grad()
            pred, loss = apply(model, criterion, batch, targets, lengths)
            loss.backward()
            optimizer.step()

            pred_idx = torch.max(pred, 1)[1]
            y_true += list(targets.int())
            y_pred += list(pred_idx.data.int())
            total_loss += loss
        acc = accuracy_score(y_true, y_pred)
        val_loss, val_acc = evaluate_validation_set(model, dev, x_to_ix, y_to_ix, criterion)
        print(
            "Train loss: {} - acc: {} \nValidation loss: {} - acc: {}".format(total_loss.data.float() / len(train), acc,
                                                                              val_loss, val_acc))
    return model


def evaluate_validation_set(model, devset, x_to_ix, y_to_ix, criterion):
    y_true = list()
    y_pred = list()
    total_loss = 0
    for batch, targets, lengths, raw_data in utils.create_dataset(devset, x_to_ix, y_to_ix, batch_size=1):
        batch, targets, lengths = utils.sort_batch(batch, targets, lengths)
        pred, loss = apply(model, criterion, batch, targets, lengths)
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int())
        total_loss += loss
    acc = accuracy_score(y_true, y_pred)
    return total_loss.data.float() / len(devset), acc


def evaluate_test_set(model, test, x_to_ix, y_to_ix):
    y_true = list()
    y_pred = list()

    for batch, targets, lengths, raw_data in utils.create_dataset(test, x_to_ix, y_to_ix, batch_size=1):
        batch, targets, lengths = utils.sort_batch(batch, targets, lengths)

        pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int())

    print(len(y_true), len(y_pred))
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))

    y_test = y_true
    y_test_pred = y_pred
    name = 'test'
    print(f'{name} set, Counter(y): {counter(y_test)}')
    print(f'cm of {name} set: ', metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred))
    report = metrics.classification_report(y_test, y_test_pred)
    # print(report)
    model.accuracy = metrics.accuracy_score(y_test, y_test_pred)
    model.F1 = metrics.f1_score(y_test, y_test_pred, average='weighted')
    model.recall = metrics.recall_score(y_test, y_test_pred, average='weighted')
    model.precision = metrics.precision_score(y_test, y_test_pred, average='weighted')
    print(f'{name} set, accuracy: {model.accuracy}, F1: {model.F1}, recall: {model.recall}, '
          f'precision: {model.precision}')


def train(args):
    random.seed(args.seed)
    data_loader = TextLoader(args.data_dir)

    train_data = data_loader.train_data
    dev_data = data_loader.dev_data
    test_data = data_loader.test_data

    char_vocab = data_loader.token2id
    tag_vocab = data_loader.tag2id
    char_vocab_size = len(char_vocab)

    print('Training samples:', len(train_data))
    print('Valid samples:', len(dev_data))
    print('Test samples:', len(test_data))

    print(char_vocab)
    print(tag_vocab)

    model = LSTMClassifier(char_vocab_size, args.char_dim, args.hidden_dim, len(tag_vocab))
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model = train_model(model, optimizer, train_data, dev_data, char_vocab, tag_vocab, args.batch_size, args.num_epochs)

    evaluate_test_set(model, test_data, char_vocab, tag_vocab)


def split_each_flow(X_train, y_train):
    new_X = []
    new_y = []

    for x, y in zip(X_train, y_train):
        new_X.append(x)
        new_y.append(y)

        d = random.randint(100, 1000)
        for i in range(int(len(x) // d)):
            sub_x = x[: (i + 1) * d]  # each subflow has "d" bytes
            new_X.append(sub_x)
            new_y.append(y)

    return new_X, new_y


def train_pcap(args):
    random.seed(args.seed)

    results = {}  # store all the results
    dim = 3000  # feature dimensions
    N_CLASSES = 8  # number of applications

    ########################################################################
    # offline train and test
    input_file = '../input_data/newapp_10220_pt.npy'
    dt = DATA(input_file, dim=dim, n_classes=N_CLASSES)  # feature length. It must be larger than 15.
    X, y = dt.sampling(n=7 * N_CLASSES, random_state=RANDOM_STATE)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)

    X_train, y_train = split_each_flow(X_train, y_train)
    X_val, y_tval = split_each_flow(X_val, y_val)
    X_test, y_test = split_each_flow(X_test, y_test)

    # tag_vocab = {'facebook': 0, 'google': 1}  # # n_classes
    tag_vocab = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}  # # n_classes
    char_vocab = {}

    train_data = list(zip(X_train, y_train))
    dev_data = list(zip(X_val, y_val))
    test_data = list(zip(X_test, y_test))

    print('Training samples:', len(train_data))
    print('Valid samples:', len(dev_data))
    print('Test samples:', len(test_data))

    print(char_vocab)
    print(tag_vocab)

    char_vocab_size = 256  # input_dim

    model = LSTMClassifier(char_vocab_size, args.char_dim, args.hidden_dim, len(tag_vocab))
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model = train_model(model, optimizer, train_data, dev_data, char_vocab, tag_vocab, args.batch_size, args.num_epochs)

    evaluate_test_set(model, test_data, char_vocab, tag_vocab)

    # model = ODCNN(in_dim=X.shape[1], epochs=20, batch_size=64, learning_rate=1e-5, n_classes=N_CLASSES)
    # model.train(X_train, y_train, X_val, y_val)
    # # evaluate on train set
    # model.test(X_train, y_train, name='train')
    # # evaluate on test set
    # model.test(X_test, y_test)

    results[0] = {'accuracy': model.accuracy, 'F1': model.F1, 'recall': model.recall, 'precision': model.precision,
                  'train_time': model.train_time, 'test_time': model.test_time}


def main(pkts_flg=1):
    if pkts_flg:
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', type=str, default='toy_data/pkts',  # default='toy_data/names'
                            help='data_directory')
        parser.add_argument('--hidden_dim', type=int, default=32,
                            help='LSTM hidden dimensions')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='size for each minibatch')
        parser.add_argument('--num_epochs', type=int, default=5,
                            help='maximum number of epochs')
        parser.add_argument('--char_dim', type=int, default=100,  # each pkt payload length
                            help='character embedding dimensions')
        parser.add_argument('--learning_rate', type=float, default=0.01,
                            help='initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-4,
                            help='weight_decay rate')
        parser.add_argument('--seed', type=int, default=123,
                            help='seed for random initialisation')
        args = parser.parse_args()
        train_pcap(args)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', type=str, default='toy_data/names',  # default='toy_data/names'
                            help='data_directory')
        parser.add_argument('--hidden_dim', type=int, default=32,
                            help='LSTM hidden dimensions')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='size for each minibatch')
        parser.add_argument('--num_epochs', type=int, default=5,
                            help='maximum number of epochs')
        parser.add_argument('--char_dim', type=int, default=128,
                            help='character embedding dimensions')
        parser.add_argument('--learning_rate', type=float, default=0.01,
                            help='initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-4,
                            help='weight_decay rate')
        parser.add_argument('--seed', type=int, default=123,
                            help='seed for random initialisation')
        args = parser.parse_args()
        train(args)


if __name__ == '__main__':
    main(pkts_flg=1)
