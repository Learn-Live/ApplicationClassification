import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import tree,svm
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import random,time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
SAMPLE= 700
marco = 8000
resample = True

random.seed(8)
#loaddata
removelist = [3,5]  # remove y label
catenumber = 8
a = np.load('newapp_10220_pt.npy')[1:]
trainY = (a[:,0]).reshape(-1,1)
trainX = (a[:,1])
truetrainX = []
truetrainY = []
for i in range(0,len(trainX)):
        if i % 1000 == 0 :
            print(i)
        Xli = trainX[i][0]
        Yli = trainY[i][0]
        containinfo = False
        #print(Yli)
        #remove certain data
        if Yli in removelist:
                continue
        elif(Yli  == 4):
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
truetrainX = np.asarray(truetrainX)
truetrainY = np.asarray(truetrainY)
print(len(truetrainX))
print("after load....")
print(truetrainX.shape)
print(truetrainY.shape)

#truncate
ss = StandardScaler()
truetrainX = ss.fit_transform(truetrainX[:,:marco])
print("after truncate....")
print(truetrainX.shape)
print(truetrainY.shape)
#display
listdir = {}
for i in range(0,len(truetrainY)):
	if truetrainY[i] not in listdir:
		listdir.update({truetrainY[i]:0})
	else:
		listdir[truetrainY[i]] = listdir[truetrainY[i]] +1
print(listdir)

#shuffle
#X_sparse = coo_matrix(truetrainX)
#truetrainX, X_sparse, truetrainY = shuffle(truetrainX, X_sparse, truetrainY, random_state=0)

#resample
listdir = {}#{[0,1,2,..400],[401,402,...,800],[801,....,1200]}
for i in range(0,len(truetrainY)):
	if truetrainY[i] not in listdir:
		listdir.update({truetrainY[i]:[i]})
	else:
		listdir[truetrainY[i]].append(i)
actualdir  = {}#{[0,2,..397],[403,...,749],[825,...1153]}
for i in range(0,10):
	if i in listdir:
		thelist = listdir[i]
	else:
		thelist = []
	if (len(thelist) > SAMPLE):
		actualdir.update({i:random.sample(thelist,SAMPLE)})#sample 500
	else:
		actualdir.update({i:thelist}) 
listdir = {}
dic = {}
truetruetrainX = []
truetruetrainY = []
for i in range(0,len(truetrainY)):
	if i not in actualdir[truetrainY[i]]:
		continue
	truetruetrainX.append(truetrainX[i])
	truetruetrainY.append(truetrainY[i])
X = np.asarray(truetruetrainX)
Y = np.asarray(truetruetrainY)
if resample == False:
	X =truetrainX  #FOR non sample
	Y = truetrainY

#resample result: XY 
'''
#modifyX 
newX = []
for item in X:
	newX=np.zeros((1,2))
	temp=np.array([item[0:5],item[5:10]],object)
	newX=np.vstack((newX,temp))
newX = np.asarray(newX)
print(newX.shape)
print(newX[0])
X = newX
'''
#input data prepare
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
listdir = {}
for i in range(0,len(y_test)):
	if y_test[i] not in listdir:
		listdir.update({y_test[i]:0})
	else:
		listdir[y_test[i]] = listdir[y_test[i]] +1
print(listdir)
listdir = {}
for i in range(0,len(y_train)):
	if y_train[i] not in listdir:
		listdir.update({y_train[i]:0})
	else:
		listdir[y_train[i]] = listdir[y_train[i]] +1
print(listdir)

#train&test
result  = []
#for i in range(10,300,30):
#	value.append(i)
#value = [100]
value = 150
print("n_estimators: ",value)
#truncatelist = [50,500,1000,1500,2000,3000,4000,5000,6000,7000,8000]
truncatelist = [50, 300,800,1500,3000,4000,6000,8000]
#for i in range(50,1500,50):
for i in truncatelist:
	print(i)
	#clf = RandomForestClassifier(n_estimators= value)
	#clf = RandomForestClassifier()
	#clf = DecisionTreeClassifier()
	#clf = SVC(kernel='linear', C=1.0)
	#clf =KNeighborsClassifier()
	#clf = GaussianNB()
	clf=  LogisticRegression(solver='lbfgs', multi_class='multinomial')
	X_train_t = X_train[:,:i]
	X_test_t = X_test[:,:i]
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
	print("train acc:",metrics.accuracy_score(y_train, predtrain))

	print(result)
print(result)