import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten,Convolution1D,Dense, Conv2D, MaxPool2D, UpSampling2D,UpSampling1D,Convolution1D,MaxPooling1D
from tensorflow.keras.models import Model,load_model
import shap
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical


#labels for two datasets
#LABELS=['Tiktok','iQiyi Video','Jindong Shopping','Snack Video','QQmusic','QQ','Taobao Shopping','NetEase Cloud Music','Arena Of Valor','WeChat']
LABELS=['chat','email','file','streaming','voip']
modelpath='./model/Proposed_classificationmodel.h5'
latent_dim = 39  #The size of the hidden layer neurons



def createDeepModel(inp_size,n_classes):
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
    encoder.summary()
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
    decoder.summary()
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
    cnn.summary()
    autoencoder=Model(input_e,decoder(encoder(input_e)))
    autoencoder.encoder=encoder
    classificationmodel= Model(input_e,cnn(encoder(input_e)))
    classificationmodel.encoder=encoder
    classificationmodel.cnn=cnn
    return autoencoder,classificationmodel




if __name__ == "__main__":


    #step1  load data
    dfDS = pd.read_csv('./dataset/'+'ISCX_5class_each_normalized_cuttedfloefeature.csv')
    featurename=dfDS.columns[1:len(dfDS.columns)]
    X_full = dfDS.iloc[:, 1:len(dfDS.columns)].values
    Y_full = dfDS["label"].values
    inp_size =X_full.shape[1]
    num_classes=len(set(Y_full))
    n_classes=len(set(Y_full))
    print("X_full",X_full.shape)
    print("n_classes",n_classes)
    x_train, x_test, y_train, y_test= train_test_split(X_full, Y_full, test_size = 0.1,random_state=5)
    x_train=x_train[0:100]
    y_train=y_train[0:100]
    x_test=x_test[0:100]
    y_test=y_test[0:100]
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    originautoencoder,originclassificationmodel =createDeepModel(inp_size,num_classes)


    # #step2 load model
    CNNmodel=load_model(modelpath)
    originclassificationmodel.set_weights(CNNmodel.get_weights())
    #pred =originclassificationmodel.predict(x_train, batch_size=64)


    #step3 Interpretable analysis of the model using shap
    explainer = shap.KernelExplainer(originclassificationmodel.predict,x_train)
    shap.initjs()
    summary_shap_values = explainer.shap_values(x_test,nsamples=80)
    shap.summary_plot(summary_shap_values,x_test,feature_names=featurename,max_display=10,plot_type='bar',class_names=LABELS)

