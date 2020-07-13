import os
import time
import numpy as np
import random
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras.callbacks
from keras import optimizers
from keras.layers import Bidirectional, LSTM, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Reshape, Dense, Dropout, \
    Activation, Flatten, BatchNormalization, LeakyReLU
from keras.utils import to_categorical, plot_model
from keras.regularizers import l2, l1, l1_l2
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras import initializers
from sklearn.utils import shuffle
import os
from scipy.stats import ks_2samp


# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


# import numpy as np
#
# np.random.seed(123456)
# x = np.random.normal(0, 1, 1000)
# y = np.random.normal(0, 1, 1000)
# z = np.random.normal(1.1, 0.9, 1000)
#
# >>> ks_2samp(x, y)
# Ks_2sampResult(statistic=0.022999999999999909, pvalue=0.95189016804849647)
# >>> ks_2samp(x, z)
# Ks_2sampResult(statistic=0.41800000000000004, pvalue=3.7081494119242173e-77)


def plot_data(history):
    import matplotlib.pyplot as plt
    plt.plot(history.acc, 'r', label='train_acc')
    plt.plot(history.valacc, 'b', label='val_acc')

    plt.plot(history.loss, 'g', label='train_loss')
    plt.plot(history.valloss, 'c', label='val_loss')

    plt.legend()
    plt.show()


# SAMPLE = 700
marco = 1500
# resample = True
# random.seed(5)
epochs = 5
batch_size = 16
catenumber = 8
ts = str(int(time.time()))
random_state = 42
lr = 10e-4
# loaddata
cate_num = 2
# http2_category = ['a','f','i','r','s','so','t','w']
# http3_category = ['g','gc','gd','gdr','gf','tb','tr','y']
if cate_num == 2:
    category_list = ['a', 'f', 'i', 'r', 's', 'so', 't', 'w']
    root_dir = '../data/http2_pi/'
    para_label = 'pi'  # g_p.txt
else:
    category_list = ['g', 'gc', 'gd', 'gdr', 'gf', 'tb', 'tr', 'y']
    # category_list = ['g']
    root_dir = '../data/all_http3/'
    para_label = 'pix'  # g_p.txt
packetlength = 19
X = []
Y = []

for cate in category_list:
    file_name = cate + "_" + para_label + ".txt"
    file_name = os.path.join(root_dir, file_name)
    print(file_name)
    with open(file_name, "r") as rfile:
        s = rfile.read()
    for line in s.split("\n"):
        nums = line.split(" ")
        if len(nums) > 1:
            truenums = [int(i) for i in nums]
            # rescale the first 19 interval time
            intval_list = truenums[0:packetlength]
            rescaled_intval_list = [int(float(i) * 255 / 100000) for i in truenums[0:packetlength]]
            truenums = rescaled_intval_list + truenums[packetlength:marco]

            X.append(truenums[0:marco])
            Y.append(category_list.index(cate))
X = np.asarray(X)
Y = np.asarray(Y)

# truncate
# truetrainX =((( truetrainX[:,:marco])/255)- 0.5)*2
# truetrainX = (truetrainX[:,:marco])/255
# ss = StandardScaler()
# truetrainX = ss.fit_transform(truetrainX[:,:marco])
# print("truetrainX1",truetrainX[0][0:20])
# print("truetrainX3",truetrainX[2][0:20])
# print("truetrainX4",truetrainX[6131][2920:2940])
# print("truetrainX5",truetrainX[4230][2920:2940])
# print("truetrainX6",truetrainX[520][2920:2940])


# for lstm
print("X:", X)
print("len 0 :", len(X[0]))
print("len 1 :", len(X[1]))
print("len 2 :", len(X[2]))
print("X.shape[0:10]", X[0][0:10])
print("X.shape[10:20]", X[0][10:20])
print("X.shape[20:30]", X[0][20:30])

print("X.shape", X.shape)
# X = X.reshape(X.shape[0],X.shape[1],1) # necessary for lstm!!

X, Y = shuffle(X, Y, random_state=random_state)
X_train1_, X_test1, y_train1_, y_test1 = train_test_split(X, Y, test_size=0.1, shuffle=True, random_state=random_state,
                                                          stratify=Y)
X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train1_, y_train1_, test_size=0.1, shuffle=True,
                                                      random_state=random_state, stratify=y_train1_)

print(X_train1.shape, y_train1.shape)
print(X_val1.shape, y_val1.shape)
print(X_test1.shape, y_test1.shape)

ss = StandardScaler()
# ss = MinMaxScaler()
# X_train = ss.fit_transform(X_train1[:, :marco])
# X_test = ss.fit_transform(X_test1[:, :marco])
# X_val = ss.fit_transform(X_val1[:, :marco])
ss.fit(X_train1[:, :marco])
X_train = ss.transform(X_train1[:, :marco])
X_val = ss.transform(X_val1[:, :marco])
X_test = ss.transform(X_test1[:, :marco])

# X_train = X_train1
# X_val = X_val1
# X_test = X_test1

y_train = to_categorical(y_train1, num_classes=catenumber)
y_val = to_categorical(y_val1, num_classes=catenumber)
y_test = to_categorical(y_test1, num_classes=catenumber)


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.valacc = []
        self.loss = []
        self.valloss = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('accuracy'))
        self.valacc.append(logs.get('val_accuracy'))
        self.loss.append(logs.get('loss'))
        self.valloss.append(logs.get('val_loss'))
        print('acc = ', self.acc)
        print('val_acc = ', self.valacc)
        print('loss = ', self.loss)
        print('val_loss = ', self.valloss)

    # def on_batch_end(self, batch, logs={}):
    #     train_loss_batch = logs.get('loss')
    #     train_acc_batch = logs.get('accuracy')
    #     # self.loss.append(train_loss_batch)
    #     self.acc.append(train_acc_batch)
    #     # print('\nTrain loss: {}, acc: {}\n'.format(train_loss_batch, train_acc_batch))
    #
    #     # x, y = self.test_data
    #     # test_loss_batch, test_acc_batch = self.model.evaluate(x, y, verbose=0)
    #     # self.valloss.append(test_loss_batch)
    #     # self.valacc.append(test_acc_batch)
    #     # print('\nTesting loss: {}, acc: {}\n'.format(test_loss_batch, test_acc_batch))


history = AccuracyHistory()
'''
#model lstm
modelname = "LSTM"
model = Sequential()
#model.add(LSTM(80))
model.add(Bidirectional(LSTM(100,activation = 'relu', input_shape=(marco, 1))))
#model.add(Dense(50,activation= 'relu'))
model.add(Dense(catenumber ,activation = 'softmax'))
'''
act = 'relu'
# act = LeakyReLU
# model cnn
# he_normal = initializers.he_normal(seed=None)
modelname = "CNN"
model = Sequential()
model.add(Reshape((marco, 1), input_shape=(marco,)))
model.add(Conv1D(8, kernel_initializer='he_normal', kernel_size=3, activation="relu"))  # 32
# model.add(Conv1D(32, kernel_size=3, activation="relu"))  # 32
# model.add(LeakyReLU(alpha=0.1))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(64, kernel_initializer='he_normal', kernel_size = 3, activation = "relu")) #16
# model.add(Conv1D(32,kernel_initializer='he_normal',  kernel_size = 3)) #16
# model.add(LeakyReLU())
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.2))
#
# #model.add(Conv1D(32, kernel_initializer='he_normal', kernel_size = 3,activation ='relu'))
# model.add(Conv1D(8, kernel_initializer='he_normal', kernel_size = 3,activation =act))
# model.add(Dropout(0.2))
# model.add(Conv1D(32, kernel_initializer='he_normal', kernel_size = 3, activation = 'relu'))
# model.add(Conv1D(8,kernel_initializer='he_normal', kernel_size=3, activation=act))
# model.add(Conv1D(8, kernel_size=3, activation='relu'))
# model.add(Conv1D(4, kernel_size=3, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.2))
# model.add(GlobalAveragePooling1D())
# model.add(Conv1D(4,kernel_initializer='he_normal', kernel_size = 3)) #16
# model.add(LeakyReLU())
# model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())  # this layer ok?
# model.add(Dense(500, kernel_initializer='he_normal', activation = 'relu'))  # activity_regularizer=l1_l2()
# model.add(LeakyReLU())
# model.add(Dropout(0.1))
# model.add(Dense(50, kernel_initializer='he_normal', activation='relu'))	#  activity_regularizer=l1_l2()
# model.add(Dense(8,kernel_initializer='he_normal',activation=act))	#  activity_regularizer=l1_l2()
# model.add(LeakyReLU())
# model.add(Dense(20, activation = 'relu'))
# model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Dense(8, kernel_initializer='he_normal', activation='softmax', activity_regularizer=l1_l2()
                ))  # activity_regularizer=l1_l2(),
# model.add(Dense(8, activation='softmax'))
'''
#model MLP
model = Sequential()
model.add(Dense(20, input_dim=marco, activation='relu'))
#model.add(Dense(200, activation='relu'))
model.add(Dense(catenumber , activation='softmax'))
'''
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=optimizers.Adam(lr=lr),
#               metrics=['accuracy'])
# sgd = optimizers.SGD(lr=10e-5, decay=1e-6, momentum=0.9, nesterov=True)
# sgd=keras.optimizers.Adam(learning_rate=10e-4, beta_1=0.9, beta_2=0.999,decay=1e-6, amsgrad=False)
sgd = keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False)
model.compile(loss=keras.losses.categorical_crossentropy,  # categorical_crossentropy
              optimizer=sgd,
              metrics=['accuracy'])
print(model.summary())
# model.fit(X_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(X_val, y_val),
#           callbacks=[history])

model.fit(X_train, y_train,  # X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          shuffle=True,
          # validation_split=0.1,
          validation_data=(X_val, y_val),
          callbacks=[history])
# print(model.summary())


# statistic parts:
timestart = time.time()
score = model.evaluate(X_test, y_test, verbose=1)
# score = model.evaluate(X_test, y_test, verbose=0)
print('score:', score)
timeend = time.time()
print("timeused: ", timeend - timestart)
y_test_pred = model.predict(X_test)
y_test_pred = np.argmax(y_test_pred, axis=1)

y_train_pred = model.predict(X_train)
y_train_pred = np.argmax(y_train_pred, axis=1)
pyname = "model_train_history" + modelname + str(marco) + ts + ".py"
# print("ts:",ts)
print(model.metrics_names)
print("score", score)
with open(pyname, "w") as wfile:
    wfile.write("import matplotlib.pyplot as plt\n")
    # wfile.write("acc = {0}\n".format(history.acc))
    # wfile.write("valacc = {0}\n".format(history.valacc))
    wfile.write("acc = {0}\n".format(history.loss))
    wfile.write("valacc = {0}\n".format(history.valloss))
    wfile.write("plt.plot(acc)\nplt.plot(valacc)\nplt.title('Model accuracy')\n"
                "plt.ylabel('Accuracy')\nplt.xlabel('Epoch')\nplt.legend(['Train', 'Test'], "
                "loc='upper left')\nplt.show()\nplt.savefig('" + modelname + str(marco) + ts + ".png')")
plot_data(history)
print(metrics.confusion_matrix(y_true=y_test1, y_pred=y_test_pred))
print(metrics.confusion_matrix(y_true=y_train1, y_pred=y_train_pred))
plot_model(model, to_file='model' + modelname + str(marco) + ts + '.png')
