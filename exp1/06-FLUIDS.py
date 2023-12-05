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
LABELS=['Tiktok','iQiyi Video','Jindong Shopping','Snack Video','QQmusic','QQ','Taobao Shopping','NetEase Cloud Music','Arena Of Valor','WeChat']
# LABELS=['chat','email','file','streaming','voip']
root_path=''
#step1  load data
dfDS = pd.read_csv('./dataset/pcapdroid_10class_each_normalized_cuttedfloefeature.csv')
X_full = dfDS.iloc[:, 1:len(dfDS.columns)].values
Y_full = dfDS["label"].values
inp_size =X_full.shape[1]
n_classes=len(set(Y_full))
print("X_full",X_full.shape)
print("n_classes",n_classes)

# define model
#encoder
input_shape=(inp_size)
input_e = Input(shape=input_shape)
x = Dense(150,activation="relu",name='dense_1')(input_e)
x = Dense(100,activation="relu",name='dense_2')(x)
x = Dense(50,activation="relu",name='dense_3')(x)
latent = Dense(latent_dim,activation="relu")(x)
encoder = Model(input_e,latent,name="encoder")
encoder.summary()
# decoder
input_d = Input(shape=(latent_dim))
x = Dense(50,activation="relu",name='dense_4')(input_d)
x = Dense(100,activation="relu",name='dense_5')(x)
x = Dense(150,activation="relu",name='dense_6')(x)
output= Dense(inp_size,activation="relu",name='dense_7')(x)
decoder = Model(input_d,output,name="decoder")
decoder.summary()
#mlp
input_c = Input(shape=(latent_dim,))
x = Dense(150,activation="relu",name='dense_8')(input_c)
x = Dense(100,activation="relu",name='dense_9')(x)
x = Dense(50,activation="relu",name='dense_10')(x)
x = Dense(n_classes,activation="softmax",name='dense_11')(x)
mlp = Model(input_c,x,name="mlp")
mlp.summary()

autoencoder=Model(input_e,decoder(encoder(input_e)))
autoencoder.encoder=encoder
classificationmodel=Model(input_e,mlp(encoder(input_e)))
classificationmodel.encoder=encoder
classificationmodel.mlp=mlp



x_train, x_test, y_train, y_test= train_test_split(X_full, Y_full, test_size = 0.1,random_state=5)
idxs_annot=getlabelindex(y_train,n_classes,labelnum) #select the labeled data

x_train_labeled   = x_train[idxs_annot]
y_train_labeled   = y_train[idxs_annot]

y_train_labeled = to_categorical(y_train_labeled)
y_test = to_categorical(y_test)
y_train=to_categorical(y_train)

# train data
classificationmodel.mlp.trainable = False # Set the encoder  to untrainable
# compile
autoencoder.compile(loss='mse',optimizer='adam',metrics='mse')
autoencoder.fit(x_train, x_train,
               epochs=100,
               batch_size=128,
               validation_data=(x_test,x_test))

classificationmodel.encoder.trainable = False # Set the encoder to untrainable
classificationmodel.mlp.trainable = True

classificationmodel.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
classificationmodel.fit(x_train_labeled, y_train_labeled ,
               epochs=100,
               batch_size=128,
               validation_data=(x_test,y_test))
encoder2weights=classificationmodel.encoder.get_weights()



#evaluate
scores = classificationmodel.evaluate(x_test,y_test , verbose=1)
y_pred =classificationmodel.mlp.predict(encoder(x_test), batch_size=100)
print("scores",scores)


#draw conf_matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test.argmax(-1),y_pred.argmax(-1))
plt.figure(figsize=(20, 20))

sns.heatmap(conf_matrix, annot=True,  fmt='d', square=True, annot_kws={"fontsize":20})
plt.title('Confusion Matrix')

plt.rcParams['font.sans-serif'] = 'simhei'

tick_marks = np.arange(len(LABELS))
plt.xticks(tick_marks, LABELS,rotation=45,fontsize=20)
plt.yticks(tick_marks, LABELS,rotation=45,fontsize=20)

plt.title("Traffic Classification Confusion Matrix (FLUIDS method)",fontsize=30)
plt.xlabel('Predicted Label',fontsize=25)
plt.ylabel('True Label',fontsize=25)
plt.savefig('Confusion Matrix_FLUIDS_6_{}_{}.png'.format(n_classes,labelnum),dpi=500)

report = classification_report(y_test.argmax(-1), y_pred.argmax(-1),target_names= LABELS,digits=4,output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv("classification_report_FLUIDS_{}.csv".format(labelnum), index=True)

