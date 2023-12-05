import time

from sklearn.preprocessing import StandardScaler
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
import tensorflow as tf


latent_dim = 40
labelnum=1000 #Set the quantity of labeled data

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

class VAE(tf.keras.Model):

    def __init__(self, latent_dim, inputsize,num_classes):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes=num_classes

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(inputsize,)),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(inputsize)
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


class preTrainTask():

    def __init__(self,latent_dim,epochs,feature=77):

        super(preTrainTask, self).__init__()
        self.latent_dim = latent_dim
        self.featuresize = feature
        self.epochs = epochs
        self.batch_size = 32


    @tf.function
    def compute_loss(self,model, x)\
            -> tf.Tensor:
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)
        reconstruction_loss = tf.reduce_mean(tf.square(x - x_logit))
        kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
        total_loss = reconstruction_loss + kl_loss

        return total_loss



    @tf.function
    def train_step(self,model, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


    def train(self,train_dataset,test_dataset,rate):

        optimizer = tf.keras.optimizers.Adam(0.000008)
        model = VAE(self.latent_dim, self.featuresize,6)
        model.encoder.summary()
        model.decoder.summary()
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            for train_x in train_dataset:
                self.train_step(model, train_x, optimizer)
            end_time = time.time()

            loss = tf.keras.metrics.Mean()
            # print('loss:', loss)
            for test_x in test_dataset:
                loss(self.compute_loss(model, test_x))
                # print('loss:', loss)
            elbo = -loss.result()
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                  .format(epoch, elbo, end_time - start_time))
        return model.encoder


    def fit(self,dataframe,rate):
        datanew=dataframe.copy()
        # testdata=testdataframe.copy()
        train, test = train_test_split(datanew, test_size=0.1, random_state=0)
        train = train.astype('float32').reshape(len(train), self.featuresize)
        test = test.astype('float32').reshape(len(test), self.featuresize)
        train_dataset = (tf.data.Dataset.from_tensor_slices(train)
                         .batch(self.batch_size))
        test_dataset = (tf.data.Dataset.from_tensor_slices(test)
                        .batch(self.batch_size))
        task = preTrainTask(latent_dim=self.latent_dim,epochs = 20,feature=self.featuresize)
        encode=task.train(train_dataset, test_dataset,rate)
        x_data_train=encode(datanew)
        return x_data_train





# LABELS=['Tiktok','iQiyi Video','Jindong Shopping','Snack Video','QQmusic','QQ','Taobao Shopping','NetEase Cloud Music','Arena Of Valor','WeChat']
LABELS=['chat','email','file','streaming','voip']
#step1  load data
dfDS = pd.read_csv('./dataset/ISCX_5class_each_normalized_cuttedfloefeature.csv')


X_full = dfDS.iloc[:, 1:len(dfDS.columns)].values


Y_full = dfDS["label"].values
inp_size =X_full.shape[1]
n_classes=len(set(Y_full))
print("X_full",X_full.shape)
print("n_classes",n_classes)


#train vae
task = preTrainTask(latent_dim=20, epochs=20, feature=66)
X_full=task.fit(X_full, 0.5)
X_full=X_full.numpy()


# define CNN
input_c = Input(shape=(latent_dim,))
x = Dense(np.prod((33,32)))(input_c)
x = Reshape((33,32))(x)
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
cnn.summary()




x_train, x_test, y_train, y_test= train_test_split(X_full, Y_full, test_size = 0.1,random_state=5)
#select labeled data
idxs_annot=getlabelindex(y_train,n_classes,labelnum)

x_train_labeled   = x_train[idxs_annot]
y_train_labeled   = y_train[idxs_annot]

y_train_labeled = to_categorical(y_train_labeled)
y_test = to_categorical(y_test)
y_train=to_categorical(y_train)

# compile model
cnn.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])
#fit cnn
cnn.fit(x_train_labeled, y_train_labeled ,
               epochs=100,
               batch_size=128,
               validation_data=(x_test, y_test))


#evaluate
scores = cnn.evaluate(x_test, y_test, verbose=1)
y_pred =cnn.predict(x_test, batch_size=100)
print("scores",scores)



#draw confusion_matrix picture
# for_10
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test.argmax(-1),y_pred.argmax(-1))
plt.figure(figsize=(20, 20))

sns.heatmap(conf_matrix, annot=True,  fmt='d', square=True, annot_kws={"fontsize":20})
plt.title('Confusion Matrix')

plt.rcParams['font.sans-serif'] = 'simhei'

tick_marks = np.arange(len(LABELS))
plt.xticks(tick_marks, LABELS,rotation=45,fontsize=20)
plt.yticks(tick_marks, LABELS,rotation=45,fontsize=20)

plt.title("Traffic Classification Confusion Matrix (VAE_CNN method)",fontsize=30)
plt.xlabel('Predicted Label',fontsize=25)
plt.ylabel('True Label',fontsize=25)
plt.savefig('Confusion Matrix_VAE_CNN method_{}_{}.png'.format(n_classes,labelnum),dpi=500)



report = classification_report(y_test.argmax(-1), y_pred.argmax(-1),target_names= LABELS,digits=4,output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv("classification_report_VAE_CNN_6_{}.csv".format(labelnum), index=True)

print(report)

