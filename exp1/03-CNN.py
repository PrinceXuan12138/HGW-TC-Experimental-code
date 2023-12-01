# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 23:29:48 2018

@author: 13913
"""

from __future__ import print_function
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D,Convolution1D
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt


labelnum=4000  #Set the quantity of labeled data
LABELS=['douyin','iqiyi','jindong','kuaishou','qqmusic','qq','taobao','wangyiyunyinyue','wangzherongyao','weixin']
#LABELS=['chat','email','file','streaming','voip']
root_path=''

#step1  Load data set
dfDS = pd.read_csv(root_path+'./dataset/'+'pcapdroid_10class_each_normalized_cuttedfloefeature.csv')
X_full = dfDS.iloc[:, 1:len(dfDS.columns)].values
Y_full = dfDS["label"].values
n_classes=len(set(Y_full))
print("n_classes",n_classes)

#step1  Load data set
x_train, x_test, y_train, y_test= train_test_split(X_full, Y_full, test_size = 0.1)
# get the dataset
inp_size =x_train.shape[1]
print("x_train",x_train.shape)
print("y_train",y_train.shape)

def getlabelindex(Y_full,n_classes,labelnum):
    Y_full=pd.DataFrame(Y_full)
    Y_full.columns=['label']
    idxs_annot=[]
    for idx in range(n_classes):
        labelindex=Y_full.loc[Y_full['label']==idx].index
        if len(labelindex)<labelnum:
            print("该类标签不足！，当前类共有标签：",len(labelindex.values),"个，但是设置抽取",labelnum,'个！！')
            idxs = np.random.choice(labelindex.values, labelnum)
        else:
            idxs = np.random.choice(labelindex.values, labelnum,replace=False)
        for data in list(idxs):
            idxs_annot.append(data)
    return idxs_annot

idxs_annot=getlabelindex(y_train,n_classes,labelnum) #Pick the labeled data

y_train = tensorflow.keras.utils.to_categorical(y_train, n_classes)
y_test  = tensorflow.keras.utils.to_categorical(y_test,  n_classes)


print("idxs_annot",idxs_annot)
print("idxs_annot",len(idxs_annot))
x_train_unlabeled = x_train
x_train_labeled   = x_train[idxs_annot]
y_train_labeled   = y_train[idxs_annot]
print("x_train_labeled",x_train_labeled.shape)
print("y_train_labeled",y_train_labeled.shape)

x_train_labeled = np.expand_dims(x_train_labeled, axis=2)
x_test = np.expand_dims(x_test, axis=2)

model=Sequential()
model.add(Convolution1D(64,3,padding="same",activation="relu",input_shape=(inp_size,1)))
model.add(Convolution1D(64,3,padding="same",activation="relu"))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Convolution1D(128,3,padding="same",activation="relu"))
model.add(Convolution1D(128,3,padding="same",activation="relu"))
model.add(MaxPool1D(pool_size=(2)))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(n_classes,activation="softmax"))
model.summary()
rmsprop=optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])


history=model.fit(x_train_labeled, y_train_labeled, epochs=100, batch_size=128,validation_split= 0.1)
model.save('./CNN/CNN_'+str(labelnum)+'_.h5')
scores = model.evaluate(x_test, y_test, verbose=1)
y_pred = model.predict(x_test, batch_size=100)



report = classification_report(y_test.argmax(-1), y_pred.argmax(-1),target_names= LABELS, digits=4)
print(report)

#step4 Draw result graph
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test.argmax(-1),y_pred.argmax(-1))
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Traffic Classification Confusion Matrix (CNN method)")
plt.ylabel('Application traffic samples')
plt.xlabel('Application traffic samples')
plt.show()

print("CNN Accuracy: ", scores[1])