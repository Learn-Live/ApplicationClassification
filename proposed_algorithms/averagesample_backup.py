import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import tree, svm
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
import random, time
from sklearn.model_selection import train_test_split

SAMPLE = 1200
marco = 10
resample = True

random.seed(5)
# loaddata
a = np.load('trdata-5bytes.npy')[1:]
trainY = (a[:, 0]).reshape(-1, 1)
trainX = (a[:, 1])
truetrainX = []
truetrainY = []
for i in range(0, len(trainX)):
    Xli = trainX[i][0]
    Yli = trainY[i][0]
    containinfo = False
    for i in Xli:
        if i != 0:
            containinfo = True
            break
    if (containinfo == True):
        truetrainX.append(Xli)
        truetrainY.append(Yli)
truetrainX = np.asarray(truetrainX)
truetrainY = np.asarray(truetrainY)
print(len(truetrainX))
print("after load....")
print(truetrainX.shape)
print(truetrainY.shape)

# truncate
truetrainX = truetrainX[:, 5:marco]

# display
listdir = {}
for i in range(0, len(truetrainY)):
    if truetrainY[i] not in listdir:
        listdir.update({truetrainY[i]: 0})
    else:
        listdir[truetrainY[i]] = listdir[truetrainY[i]] + 1
print(listdir)

# shuffle
# X_sparse = coo_matrix(truetrainX)
# truetrainX, X_sparse, truetrainY = shuffle(truetrainX, X_sparse, truetrainY, random_state=0)

# resample
listdir = {}  # {[0,1,2,..400],[401,402,...,800],[801,....,1200]}
for i in range(0, len(truetrainY)):
    if truetrainY[i] not in listdir:
        listdir.update({truetrainY[i]: [i]})
    else:
        listdir[truetrainY[i]].append(i)
actualdir = {}  # {[0,2,..397],[403,...,749],[825,...1153]}
for i in range(0, 10):
    if i in listdir:
        thelist = listdir[i]
    else:
        thelist = []
    if (len(thelist) > SAMPLE):
        actualdir.update({i: random.sample(thelist, SAMPLE)})  # sample 500
    else:
        actualdir.update({i: thelist})
listdir = {}
dic = {}
truetruetrainX = []
truetruetrainY = []
for i in range(0, len(truetrainY)):
    if i not in actualdir[truetrainY[i]]:
        continue
    truetruetrainX.append(truetrainX[i])
    truetruetrainY.append(truetrainY[i])
X = np.asarray(truetruetrainX)
Y = np.asarray(truetruetrainY)
if resample == False:
    X = truetrainX  # FOR non sample
    Y = truetrainY

# resample result: XY
# modifyX
newX = []
for item in X:
    print(item.shape)
    print(item[0:5])
    print(item[5:marco])
    newX.append([item[0:5], item[5:marco]])
    time.sleep(5)
newX = np.asarray(newX)
print(newX.shape)
print(newX[0])
# input data prepare
X_train, X_test, y_train, y_test = train_test_split(newX, Y, test_size=0.1, random_state=42)
listdir = {}
for i in range(0, len(y_test)):
    if y_test[i] not in listdir:
        listdir.update({y_test[i]: 0})
    else:
        listdir[y_test[i]] = listdir[y_test[i]] + 1
print(listdir)
listdir = {}
for i in range(0, len(y_train)):
    if y_train[i] not in listdir:
        listdir.update({y_train[i]: 0})
    else:
        listdir[y_train[i]] = listdir[y_train[i]] + 1
print(listdir)

# train&test
result = []
# for i in range(10,300,30):
#	value.append(i)
# value = [100]
value = 150
print("n_estimators: ", value)
truncatelist = [55, 205, 805, 1505, 3000]
# for i in range(50,1500,50):
for i in truncatelist:
    print(i)
    clf = RandomForestClassifier(n_estimators=value)
    X_train_t = X_train[:, :i]

    X_test_t = X_test[:, :i]
    print("before input....")
    print(X_train_t.shape)
    print(y_train.shape)
    print(X_test_t.shape)
    print(y_test.shape)
    print((X_train_t[0])[0:10])
    clf.fit(X_train_t, y_train)
    predtest = clf.predict(X_test_t)
    result.append(metrics.accuracy_score(y_test, predtest))
    print(confusion_matrix(y_test, predtest))
    predtrain = clf.predict(X_train_t)
    print(confusion_matrix(y_train, predtrain))
    print("train acc:", metrics.accuracy_score(y_train, predtrain))

    print(result)
print(result)
