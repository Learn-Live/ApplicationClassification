import os
import time
import numpy as np
import random
from keras.models import Sequential
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
from sklearn.tree import DecisionTreeClassifier
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
epochs = 10
batch_size = 10
catenumber = 8
ts = str(int(time.time()))
random_state = 42
lr = 10e-4
# loaddata
cate_num = 3
# http2_category = ['a','f','i','r','s','so','t','w']
# http3_category = ['g','gc','gd','gdr','gf','tb','tr','y']
if cate_num == 2:
    category_list = ['a', 'f', 'i', 'r', 's', 'so', 't', 'w']
else:
    category_list = ['g', 'gc', 'gd', 'gdr', 'gf', 'tb', 'tr', 'y']
    category_list = ['g', 'tb', 'tr', 'y']
# category_list = ['g']
para_label = 'pix'  # g_p.txt
packetlength = 19
X = []
Y = []
X_train = []
y_train = []
X_test = []
y_test = []
X_val = []
y_val = []
root_dir = '../data/all_http3/'
for cate in category_list:
    file_name = cate + "_" + para_label + ".txt"
    file_name = os.path.join(root_dir, file_name)
    print(file_name)
    with open(file_name, "r") as rfile:
        s = rfile.read()

    i = 0
    for line in s.split("\n"):
        nums = line.split(" ")
        if len(nums) > 1:
            truenums = [int(i) for i in nums]
            # rescale the first 19 interval time
            intval_list = truenums[0:packetlength]
            # rescaled_intval_list = [int(float(i) * 255 / 100000) for i in truenums[0:packetlength]]
            rescaled_intval_list = [float(i) * 255 / 100000 for i in truenums[0:packetlength]]
            truenums = rescaled_intval_list + truenums[packetlength:marco]

            X.append(truenums[0:marco])
            Y.append(category_list.index(cate))

            if i % 2 == 0:  # i < 800:
                X_train.append(truenums[0:marco])
                y_train.append(category_list.index(cate))
            elif i % 5 == 0 and i % 2 != 0:  # i >= 800 and i < 1000:
                X_test.append(truenums[0:marco])
                y_test.append(category_list.index(cate))
            else:
                X_val.append(truenums[0:marco])
                y_val.append(category_list.index(cate))
            i += 1

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

# X,Y =shuffle(X, Y, random_state=random_state)
# # X_train1_, X_test1, y_train1_, y_test1 = train_test_split(X, Y, test_size=0.1,shuffle=True, random_state=random_state, stratify=Y)
# # X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train1_, y_train1_, test_size=0.1, shuffle=True, random_state=random_state,  stratify=y_train1_)
# X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y, test_size=0.3, random_state=random_state)
# X_val1 = X_test1
# y_val1 = y_test1
# print(X_train1.shape, y_train1.shape)
# print(X_val1.shape, y_val1.shape)
# print(X_test1.shape, y_test1.shape)
#
# ss = StandardScaler()
# # ss = MinMaxScaler()
# # X_train = ss.fit_transform(X_train1[:, :marco])
# # X_test = ss.fit_transform(X_test1[:, :marco])
# # X_val = ss.fit_transform(X_val1[:, :marco])
# ss.fit(X_train1[:, :marco])
# X_train = ss.transform(X_train1[:, :marco])
# X_val = ss.transform(X_val1[:, :marco])
# X_test = ss.transform(X_test1[:, :marco])

# X_train = X_train1
# X_val = X_val1
# X_test = X_test1

# y_train = to_categorical(y_train1, num_classes=catenumber)
# y_val = to_categorical(y_val1, num_classes=catenumber)
# y_test = to_categorical(y_test1, num_classes=catenumber)
# y_train = y_train1
# y_val = y_val1
# y_test = y_test1

dt = DecisionTreeClassifier(min_samples_leaf=10)
# dt = RandomForestClassifier(min_samples_leaf=100, max_depth=5)
# dt.fit(X_train, y_train)
# dt.fit(X_test, y_test)
# print(cross_val_score(dt, X_train, y_train, cv=5))
#
scores = model_selection.cross_validate(dt, X_train, y_train, cv=3, return_train_score=True, return_estimator=True)
print('Train scores:')
print(scores['train_score'])
print('Test scores:')
print(scores['test_score'])
max_i = np.argmax(scores['test_score'])
dt = scores['estimator'][max_i]

#
# parameters = {'max_depth':range(3,20)}
# clf = GridSearchCV(dt, parameters, n_jobs=4)
# clf.fit(X=X_train, y=y_train)
# dt = clf.best_estimator_
# print(clf.best_score_, clf.best_params_)

y_train_pred = dt.predict(X_train)
print('x_train:', dt.score(X_train, y_train))
y_test_pred = dt.predict(X_test)
print('x_test:', dt.score(X_test, y_test))

print(metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred))
print(metrics.confusion_matrix(y_true=y_train, y_pred=y_train_pred))
# plot_model(model, to_file='model' + modelname + str(marco) + ts + '.png')


gs_cv = GridSearchCV(dt, cv=3, param_grid={'min_samples_leaf': [1, 2, 5, 10]}, scoring='accuracy')
gs_cv.fit(X_train, y_train)
print(gs_cv.best_score_)
print(gs_cv.best_index_)
print(gs_cv.best_params_)
dt = gs_cv.best_estimator_

#
# parameters = {'max_depth':range(3,20)}
# clf = GridSearchCV(dt, parameters, n_jobs=4)
# clf.fit(X=X_train, y=y_train)
# dt = clf.best_estimator_
# print(clf.best_score_, clf.best_params_)

y_train_pred = dt.predict(X_train)
print('x_train:', dt.score(X_train, y_train))
y_test_pred = dt.predict(X_test)
print('x_test:', dt.score(X_test, y_test))

print(metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred))
print(metrics.confusion_matrix(y_true=y_train, y_pred=y_train_pred))
# plot_model(model, to_file='model' + modelname + str(marco) + ts + '.png')
