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

        # # multilabels
        # new_y = [[]]*len(y)
        # for i, _y_i in enumerate(y):
        #     mapping={'chat':0, 'video':1, 'streaming':2, 'games':3}
        #     if _y_i == 0:
        #         new_y[i] = [_y_i, mapping['chat'], 0, 0, 0]
        #     elif _y_i == 1 :
        #         new_y[i] = [_y_i, mapping['chat'], mapping['video'], 0,0]
        #     elif _y_i == 2:
        #         new_y[i] = [_y_i, mapping['chat'],mapping['video'], mapping['streaming'], 0]
        #     elif _y_i == 3:
        #         new_y[i] = [_y_i, 0, mapping['video'], mapping['streaming'], mapping['games']]
        #     elif _y_i == 4:
        #         new_y[i] = [_y_i, 0, 0, mapping['streaming'], mapping['games']]
        #     elif _y_i == 5:
        #         new_y[i] = [_y_i, 0,0,0, mapping['games']]
        #     elif _y_i == 6:
        #         new_y[i] = [_y_i, mapping['chat'], 0, mapping['streaming'], mapping['games']]
        #     elif _y_i == 7:
        #         new_y[i] = [_y_i, mapping['chat'], 0,0 , mapping['games']]
        #     else:
        #         new_y[i] = [_y_i, 0, 0,0,0]

        # multilabels
        new_y = [[]] * len(y)
        for i, _y_i in enumerate(y):
            mapping = {'chat': 0, 'video': 1, 'streaming': 2, 'games': 3}
            if _y_i == 0:
                new_y[i] = [_y_i, mapping['chat']]
            elif _y_i == 1:
                new_y[i] = [_y_i, mapping['video']]
            elif _y_i == 2:
                new_y[i] = [_y_i, mapping['chat']]
            elif _y_i == 3:
                new_y[i] = [_y_i, mapping['video']]
            elif _y_i == 4:
                new_y[i] = [_y_i, mapping['games']]
            elif _y_i == 5:
                new_y[i] = [_y_i, mapping['streaming']]
            elif _y_i == 6:
                new_y[i] = [_y_i, mapping['streaming']]
            elif _y_i == 7:
                new_y[i] = [_y_i, mapping['games']]
            else:
                new_y[i] = [_y_i, 0]

        y = new_y

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
            model.add(Dense(self.n_classes + 4, activation='softmax', activity_regularizer=l1_l2()))
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

        # for y_train
        y0 = [y0 for y0, y1 in y_train]
        y0 = to_categorical(y0, num_classes=self.n_classes)

        y1 = [y1 for y0, y1 in y_train]
        y1 = to_categorical(y1, num_classes=4)
        y_train = np.asarray([np.asarray(list(v0) + list(v1), dtype=int) for v0, v1 in zip(y0, y1)], dtype=int)

        # for y_val
        y0 = [y0 for y0, y1 in y_val]
        y0 = to_categorical(y0, num_classes=self.n_classes)

        y1 = [y1 for y0, y1 in y_val]
        y1 = to_categorical(y1, num_classes=4)
        y_val = np.asarray([np.asarray(list(v0) + list(v1), dtype=int) for v0, v1 in zip(y0, y1)], dtype=int)

        # y_val = to_categorical(y_val, num_classes=self.n_classes)

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
        y_test_pred_0 = [v[:8] for v in y_test_pred]
        y_test_pred_1 = [v[8:] for v in y_test_pred]
        for i in range(2):
            print(f'===predict label {i}')
            y_test_pred = y_test_pred_0 if i == 0 else y_test_pred_1
            y_test_true = [v[0] for v in y_test] if i == 0 else [v[1] for v in y_test]
            y_test_pred = np.argmax(y_test_pred, axis=1)

            timeend = time.time()
            self.test_time = timeend - timestart
            print(f"Testing time on {name} set: ", self.test_time)

            # print(f'{name} set, Counter(y): {counter(y_test)}')
            print(f'cm of {name} set: ', metrics.confusion_matrix(y_true=y_test_true, y_pred=y_test_pred))
            report = metrics.classification_report(y_test_true, y_test_pred)
            # print(report)
            self.accuracy = metrics.accuracy_score(y_test_true, y_test_pred)
            self.F1 = metrics.f1_score(y_test_true, y_test_pred, average='weighted')
            self.recall = metrics.recall_score(y_test_true, y_test_pred, average='weighted')
            self.precision = metrics.precision_score(y_test_true, y_test_pred, average='weighted')
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

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_val = ss.fit_transform(X_val)
    X_test = ss.fit_transform(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def main():
    results = {}  # store all the results
    dim = 3000  # feature dimensions
    N_CLASSES = 8  # number of applications
    epochs = 20
    ########################################################################
    # offline train and test
    input_file = '../input_data/newapp_10220_pt.npy'
    dt = DATA(input_file, dim=dim, n_classes=N_CLASSES)  # feature length. It must be larger than 15.
    X, y = dt.sampling(n=700 * N_CLASSES, random_state=RANDOM_STATE)

    model = ODCNN(in_dim=X.shape[1], epochs=epochs, batch_size=64, learning_rate=1e-5, n_classes=N_CLASSES)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)
    model.train(X_train, y_train, X_val, y_val)
    # evaluate on train set
    model.test(X_train, y_train, name='train')
    # evaluate on test set
    model.test(X_test, y_test)

    results[0] = {'accuracy': model.accuracy, 'F1': model.F1, 'recall': model.recall, 'precision': model.precision,
                  'train_time': model.train_time, 'test_time': model.test_time}
    ########################################################################
    # Retrain and online test
    retrain_online_test = True
    cnt = 1
    while retrain_online_test and cnt <= 1:
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
