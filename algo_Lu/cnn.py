import numpy as np
import random, time
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras.callbacks
from keras import optimizers
from keras.layers import Bidirectional, LSTM, Conv1D, MaxPooling1D, GlobalAveragePooling1D,Reshape, Dense, Dropout, Activation, Flatten, BatchNormalization, LeakyReLU
from keras.utils import to_categorical, plot_model
from keras.regularizers import l2, l1, l1_l2
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
SAMPLE= 700
marco = 44
resample = True
#random.seed(5)
epochs = 100
batch_size = 64
ts = str(int(time.time()))
#loaddata
print("loading data.....")
removelist = [3,5]  # remove y label
catenumber = 8
a = np.load('newapp_2000_t.npy')[1:]
trainY = (a[:,0]).reshape(-1,1)
trainX = (a[:,1])
truetrainX = []
truetrainY = []
for i in range(0,len(trainX)):
	if i % 1000 == 0:
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
#truetrainX =((( truetrainX[:,:marco])/255)- 0.5)*2
#truetrainX = (truetrainX[:,:marco])/255
#ss = StandardScaler()
#truetrainX = ss.fit_transform(truetrainX[:,:marco])
#print("truetrainX1",truetrainX[0][0:20])
print("truetrainX2",truetrainX[1][0:20])
#print("truetrainX3",truetrainX[2][0:20])
#print("truetrainX4",truetrainX[6131][2920:2940])
#print("truetrainX5",truetrainX[4230][2920:2940])
#print("truetrainX6",truetrainX[520][2920:2940])
#resample
listdir = {}
for i in range(0,len(truetrainY)):
	if truetrainY[i] not in listdir:
		listdir.update({truetrainY[i]:[i]})
	else:
		listdir[truetrainY[i]].append(i)
actualdir  = {}
for i in range(0,catenumber ):
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
	X = truetrainX  #FOR non sample
	Y = truetrainY
#for lstm
print("X.shape[0]",X.shape[0])
print("X.shape[1]",X.shape[1])
#X = X.reshape(X.shape[0],X.shape[1],1) # necessary for lstm!!


X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y, test_size=0.1, random_state=42)
X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train1, y_train1, test_size=0.1, random_state=5)

ss = StandardScaler()
X_train = ss.fit_transform(X_train1[:,:marco])
X_test = ss.fit_transform(X_test1[:,:marco])
X_val = ss.fit_transform(X_val1[:,:marco])



y_train = to_categorical(y_train1, num_classes=catenumber )
y_test = to_categorical(y_test1, num_classes=catenumber )
y_val = to_categorical(y_val1, num_classes=catenumber )

class AccuracyHistory(keras.callbacks.Callback):
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
        print('acc = ',self.acc)
        print('val_acc = ',self.valacc)
        print('loss = ',self.loss)
        print('val_loss = ',self.valloss)
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
#model cnn
modelname = "CNN"
model = Sequential()
model.add(Reshape((marco, 1), input_shape=(marco,)))
model.add(Conv1D(16, strides = 1, kernel_size = 3, activation = "relu")) #32
#model.add(LeakyReLU(alpha=0.1))
#model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(16,strides = 1, kernel_size = 3, activation = "relu")) #16
#model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv1D(32, kernel_size = 3,activation ='relu'))
model.add(Conv1D(32, kernel_size = 3, activation = 'relu')) 
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
#model.add(GlobalAveragePooling1D())
#model.add(Conv1D(8,strides =1, kernel_size = 3)) #16
#model.add(LeakyReLU(alpha=0.1))
#model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())#this layer ok?
model.add(Dense(500, activation = 'relu'))
#model.add(LeakyReLU())
model.add(Dense(300, activation = 'relu'))
#model.add(LeakyReLU())
#model.add(Dense(20, activation = 'relu'))
#model.add(LeakyReLU())
model.add(Dense(8, activation='softmax',activity_regularizer=l1_l2()))
#model.add(Dense(8, activation='softmax'))
'''
#model MLP
model = Sequential()
model.add(Dense(20, input_dim=marco, activation='relu'))
#model.add(Dense(200, activation='relu'))
model.add(Dense(catenumber , activation='softmax'))
'''
model.compile(loss=keras.losses.categorical_crossentropy,
	optimizer=optimizers.Adam(lr=0.0005 ),
	metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train,
	batch_size=batch_size,
	epochs = epochs,
	verbose=1,
	validation_data=(X_val, y_val),
	callbacks=[history])
#print(model.summary())
# statistic parts:
timestart = time.time()
score = model.evaluate(X_test, y_test, verbose=0)
timeend = time.time()
print("timeused: ", timeend  - timestart)
y_test_pred = model.predict(X_test)
y_test_pred = np.argmax(y_test_pred , axis=1)
y_train_pred = model.predict(X_train)
y_train_pred = np.argmax(y_train_pred , axis=1)
pyname = "model_train_history"+modelname+str(marco)+ts+".py"
#print("ts:",ts)
print(model.metrics_names)
print("score", score)
with open(pyname,"w") as wfile:
	wfile.write("import matplotlib.pyplot as plt\n")
	wfile.write("acc = {0}\n".format(history.acc))
	wfile.write("valacc = {0}\n".format(history.valacc))
	wfile.write("plt.plot(acc)\nplt.plot(valacc)\nplt.title('Model accuracy')\nplt.ylabel('Accuracy')\nplt.xlabel('Epoch')\nplt.legend(['Train', 'Test'], loc='upper left')\nplt.savefig('"+modelname+str(marco)+ts+".jpg')")

print(metrics.confusion_matrix(y_true=y_test1, y_pred=y_test_pred))
print(metrics.confusion_matrix(y_true=y_train1, y_pred=y_train_pred))
plot_model(model, to_file='model'+modelname+str(marco)+ts+'.png')
