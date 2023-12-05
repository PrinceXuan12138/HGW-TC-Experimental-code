from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten,Convolution1D,Dense, Conv2D, MaxPool2D, UpSampling2D,UpSampling1D,Convolution1D,MaxPooling1D
from tensorflow.keras.models import Model,load_model
from tensorflow.keras import backend as K
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np


latent_dim = 39  #The size of the hidden layer neurons
labelnum=4000   #The amount of labeled data (per class)

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




#labels for two datasets
#LABELS=['Tiktok','iQiyi Video','Jindong Shopping','Snack Video','QQmusic','QQ','Taobao Shopping','NetEase Cloud Music','Arena Of Valor','WeChat']
LABELS=['chat','email','file','streaming','voip']
root_path=''
#step1  load data
dfDS = pd.read_csv(root_path+'./dataset/'+'ISCX_5class_each_normalized_cuttedfloefeature.csv')
X_full = dfDS.iloc[:, 1:len(dfDS.columns)].values
Y_full = dfDS["label"].values
inp_size =X_full.shape[1]
n_classes=len(set(Y_full))
print("X_full",X_full.shape)
print("n_classes",n_classes)

# define model
#encoder
input_shape=(inp_size,1)
input_e = Input(shape=input_shape)
x =Convolution1D(64,3,padding="same",activation="relu",name='conv_1')(input_e)
x =Convolution1D(64,3,padding="same",activation="relu",name='conv_2')(x)
x=MaxPooling1D(name='maxpool_1')(x)
x=Convolution1D(32,3,padding="same",activation="relu",name='conv_3')(x)
x=Convolution1D(32,3,padding="same",activation="relu",name='conv_4')(x)
s_shape = K.int_shape(x)
x = Flatten()(x)
latent = Dense(latent_dim,activation="relu")(x)
encoder = Model(input_e,latent,name="encoder")
# encoder.summary()
# decoder
input_d = Input(shape=(latent_dim))
x = Dense(np.prod(s_shape[1:]))(input_d)
x = Reshape((s_shape[1],s_shape[2]))(x)
x = Convolution1D(32,3,padding="same",activation="relu",name='conv_5')(x)
x = Convolution1D(32,3,padding="same",activation="relu",name='conv_6')(x)
x = UpSampling1D(name='upsampling_1d_2')(x)
x = Convolution1D(64,3,padding="same",activation="relu",name='conv_7')(x)
output= Convolution1D(1,3,padding="same",activation="relu",name='conv_8')(x)
decoder = Model(input_d,output,name="decoder")
# decoder.summary()
#CNN
input_c = Input(shape=(latent_dim,))
x = Dense(np.prod(s_shape[1:]))(input_c)
x = Reshape((s_shape[1],s_shape[2]))(x)
x = Convolution1D(64,3,padding="same",activation="relu")(x)
x = Convolution1D(64,3,padding="same",activation="relu")(x)
x = MaxPooling1D(pool_size=(2))(x)
x = Convolution1D(128,3,padding="same",activation="relu")(x)
x = Convolution1D(128,3,padding="same",activation="relu")(x)
x = MaxPooling1D(pool_size=(2))(x)
x = Flatten()(x)
x = Dense(128,activation="relu")(x)
x = Dense(n_classes,activation="softmax")(x)
cnn = Model(input_c,x,name="AEcnn")

autoencoder=Model(input_e,decoder(encoder(input_e)))
autoencoder.encoder=encoder
classificationmodel=Model(input_e,cnn(encoder(input_e)))
classificationmodel.encoder=encoder
classificationmodel.cnn=cnn
totalmodel= Model(input_e,[cnn(encoder(input_e)),decoder(encoder(input_e))])
totalmodel.autoencoder=autoencoder
totalmodel.classificationmodel=classificationmodel


x_train, x_test, y_train, y_test= train_test_split(X_full, Y_full, test_size = 0.1,random_state=5)

idxs_annot=getlabelindex(y_train,n_classes,labelnum) #select the labeled data
x_train_labeled   = x_train[idxs_annot]
y_train_labeled   = y_train[idxs_annot]

y_train_labeled = to_categorical(y_train_labeled)
y_test = to_categorical(y_test)



# train data
totalmodel.classificationmodel.cnn.trainable = False #  Set the encoder  to untrainable
# compile
totalmodel.autoencoder.compile(loss='mse',optimizer='adam',metrics='mse')
totalmodel.autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                validation_data=(x_test,x_test))

totalmodel.classificationmodel.encoder.trainable = False # Set the encoder to untrainable
totalmodel.classificationmodel.cnn.trainable = True

totalmodel.classificationmodel.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
totalmodel.classificationmodel.fit(x_train_labeled, y_train_labeled,
                        epochs=50,
                        batch_size=128,
                        validation_data=(x_test,y_test))


classificationmodel.save('./model/Proposed_classificationmodel.h5')



#evaluate
totalmodel.compile(optimizer="adam",loss=["categorical_crossentropy","mse"],metrics=["accuracy"])
scores = totalmodel.evaluate(x_test, [y_test,x_test], verbose=1)
y_pred =totalmodel.classificationmodel.cnn.predict(encoder(x_test), batch_size=100)
print("scores",scores)

#draw conf_matrix
# # for_10
# from sklearn.metrics import confusion_matrix
# conf_matrix = confusion_matrix(y_test.argmax(-1),y_pred.argmax(-1))
# plt.figure(figsize=(20, 20))

# sns.heatmap(conf_matrix, annot=True,  fmt='d', square=True, annot_kws={"fontsize":20})
# plt.title('Confusion Matrix')

# plt.rcParams['font.sans-serif'] = 'simhei'

# tick_marks = np.arange(len(LABELS))
# plt.xticks(tick_marks, LABELS,rotation=45,fontsize=20)
# plt.yticks(tick_marks, LABELS,rotation=45,fontsize=20)
# plt.title("Traffic Classification Confusion Matrix (AECNN method)",fontsize=30)
# plt.xlabel('Predicted Label',fontsize=25)
# plt.ylabel('True Label',fontsize=25)
# plt.savefig('Confusion Matrix_Proposed_10_{}.png'.format(labelnum),dpi=500)
# plt.show()


#for_6
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test.argmax(-1),y_pred.argmax(-1))
plt.figure(figsize=(20, 20))
sns.heatmap(conf_matrix, annot=True,  fmt='d', square=True, annot_kws={"fontsize":20})
plt.title('Confusion Matrix')
plt.rcParams['font.sans-serif'] = 'simhei'
tick_marks = np.arange(len(LABELS))
plt.xticks(tick_marks, LABELS,rotation=45,fontsize=20)
plt.yticks(tick_marks, LABELS,rotation=45,fontsize=20)
plt.title("Traffic Classification Confusion Matrix (AECNN method)",fontsize=30)
plt.xlabel('Predicted Label',fontsize=25)
plt.ylabel('True Label',fontsize=25)
plt.savefig('Confusion Matrix_proposed_6_{}.png'.format(labelnum),dpi=500)

report = classification_report(y_test.argmax(-1), y_pred.argmax(-1),target_names= LABELS,digits=4,output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv("classification_report_Proposed_{}.csv".format(labelnum), index=True)



# model=load_model('./Models/totalmodel.h5')
# print(model)
# model.test_model.summary()
# m = keras.models.clone_model(model)
# totalmodel.load_model('./Models/totalmodel.h5')