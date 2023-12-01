# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 23:29:48 2018

@author: 13913
"""
from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import time
import datetime
import os
import psutil

from sklearn.model_selection import train_test_split
from threading import Thread
import GPUtil
import csv
import tensorflow.keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten,Convolution1D,Dense, Conv2D, MaxPool2D, UpSampling2D,UpSampling1D,Convolution1D,MaxPooling1D
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


#LABELS=['douyin','iqiyi','jindong','kuaishou','qqmusic','qq','taobao','wangyiyunyinyue','wangzherongyao','weixin']
LABELS=['chat','email','file','streaming','voip']
root_path=''
labelnum=1500  #每个类有标签的数据量
latent_dim = 39
#每个模型的参数
verbose, epochs, batch_size =2,5, 64
numOfIterations=50    #全局训练轮数
numOfClients=10   #子节点个数
monitoring_filename= "./result/联邦学习AECNN_"+str(labelnum)+"label_性能监控.csv"
performance_filename= "./result/联邦学习AECNN_"+str(labelnum)+"label_评价指标.csv"
AEmodelLocation="./Models/FL_AE_"+str(numOfClients)+"_nodes.h5"
CNNmodelLocation="./Models/FL_CNN_"+str(numOfClients)+"_nodes.h5"

########CNN相关##########
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



def splitLabel(x_train,y_train):
    idxs_annot = np.random.choice(x_train.shape[0], labelnum)
    x_train_labeled   = x_train[idxs_annot]
    y_train_labeled   = y_train[idxs_annot]
    x_train_unlabeled=x_train_labeled
    return  x_train_labeled,y_train_labeled,x_train_unlabeled


#######深度学习相关######
accList, precList, recallList, f1List = [], [], [], []
deepAEModelAggWeights=[]
deepCNNModelAggWeights=[]
firstClientFlag=True
def trainInServer(model,x_train_labeled,y_train_labeled):
    model.compile(optimizer="adam",loss=["categorical_crossentropy","mse"],metrics=["accuracy"])
    model.fit(x_train_labeled, [y_train_labeled,x_train_labeled],
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True,
              validation_split= 0.3,verbose=verbose)

def updateServerModel(clientAEWeight,clientCNNWeight):
    global firstClientFlag
    for ind in range(len(clientAEWeight)):
        if(firstClientFlag==True):
            deepAEModelAggWeights.append(clientAEWeight[ind])
            deepCNNModelAggWeights.append(clientCNNWeight[ind])
        else:
            deepAEModelAggWeights[ind]=(deepAEModelAggWeights[ind]+clientAEWeight[ind])
            deepCNNModelAggWeights[ind]=(deepCNNModelAggWeights[ind]+clientCNNWeight[ind])


def updateClientsModels():
    global clientsAEModelList
    global clientsCNNModelist
    global originautoencoder
    global originclassificationmodel
    clientsAEModelList.clear()
    clientsCNNModelist.clear()

    for clientID in range(numOfClients):
        updatedAEmodel = tensorflow.keras.models.clone_model(originautoencoder)
        updatedAEmodel.set_weights(originautoencoder.get_weights())
        clientsAEModelList.append(updatedAEmodel)

        updatedCNNmodel = tensorflow.keras.models.clone_model(originclassificationmodel)
        updatedCNNmodel.set_weights(originclassificationmodel.get_weights())
        clientsCNNModelist.append(updatedCNNmodel)




############统计相关###########
monitorheaders = ['stage','iterationNo','clientID','avg_GPU_mem','avg_GPU_load','avg_Memory_used','avg_cpu_used','used_time(us)']
Globalmonitordirct={'stage':'','iterationNo':0,'clientID':0,'avg_GPU_mem':0,'avg_GPU_load':0,'avg_Memory_used':0,'avg_cpu_used':0,'used_time(us)':0}
Globalmonitordirctrows=[]

performanceheaders=['stage','iterationNo','clientID','train_acc','val_acc','test_acc','classification_report']
performancerdirct={'stage':'','iterationNo':0,'clientID':0,'train_acc':0,'val_acc':0,'test_acc':0,'classification_report':''}
performancerdirctros=[]
#代码性能监控
class Monitor(Thread):
    def __init__(self, delay,stage,iterationNo,clientID,process):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.stage = stage
        self.iterationNo = iterationNo
        self.clientID = clientID
        self.process=process
        self.gpu_mem_list=[]
        self.gpu_load_list=[]
        self.used_mem_list=[]
        self.cpu_load_list=[]


        self.start()
    def run(self):
        starttesttime=datetime.datetime.now()
        while not self.stopped:
            # GPUtil.showUtilization()
            Gpus = GPUtil.getGPUs()
            gpu=Gpus[0]
            self.gpu_mem_list.append(gpu.memoryUtil * 100)
            self.gpu_load_list.append(gpu.load * 100)
            self.used_mem_list.append(process.memory_percent(memtype="uss"))
            self.cpu_load_list.append(process.cpu_percent(interval = 1))
            time.sleep(self.delay)
        endtesttime = datetime.datetime.now()
        used_time=(endtesttime-starttesttime).microseconds
        Globalmonitordirct['stage']=self.stage
        Globalmonitordirct['iterationNo']=self.iterationNo
        Globalmonitordirct['clientID']=self.clientID
        Globalmonitordirct['avg_GPU_mem']=np.mean(self.gpu_mem_list)
        Globalmonitordirct['avg_GPU_load']=np.mean(self.gpu_load_list)
        Globalmonitordirct['avg_Memory_used']=np.mean(self.used_mem_list)
        Globalmonitordirct['avg_cpu_used']=np.mean(self.cpu_load_list)
        Globalmonitordirct['used_time(us)']=used_time*23
        Globalmonitordirctrows.append(Globalmonitordirct.copy())


    def stop(self):
        self.stopped = True
        return self.gpu_mem_list,self.gpu_load_list


if __name__ == '__main__':
    #step1  加载数据集
    dfDS = pd.read_csv(root_path+'./dataset/'+'ISCX_5class_each_normalized_cuttedfloefeature.csv')
    X_full = dfDS.iloc[:, 1:len(dfDS.columns)].values
    Y_full = dfDS["label"].values
    num_classes=len(set(Y_full))
    Y_full = tensorflow.keras.utils.to_categorical(Y_full, num_classes)

    # FOR TEST SPLIT
    xServer, xClients, yServer, yClients = train_test_split(X_full, Y_full, test_size=0.90,random_state=523)
    print("yServer",yServer.shape)
    print("yClients",yClients.shape)
    xServer = np.expand_dims(xServer, axis=2)

    #创建初始模型
    inp_size = xServer.shape[1]
    originautoencoder,originclassificationmodel =createDeepModel(inp_size,num_classes)
    originautoencoder.save(AEmodelLocation)
    originclassificationmodel.save(CNNmodelLocation)

    # ----- 1. 训练初始模型 -----
    # x_train_labeled,y_train_labeled=splitLabel(xServer,yServer)
    # x_train_labeled = np.expand_dims(x_train_labeled, axis=2)
    # servermodel=initialtotalmodel
    #trainInServer(servermodel,x_train_labeled,y_train_labeled)


    # ------- 2. 拆分子节点训练数据 ----------
    xClientsList=[]
    yClientsList=[]
    xClientsListLabel=[]
    xClientsListUnLabel=[]
    yClientsListLabel=[]

    clientsAEModelList=[]
    clientsCNNModelist=[]

    clientDataInterval=len(xClients)//numOfClients
    lastLowerBound=0


    for clientID in range(numOfClients):
        xClientsList.append(xClients[lastLowerBound : lastLowerBound+clientDataInterval])
        yClientsList.append(yClients[lastLowerBound : lastLowerBound+clientDataInterval])
        AEmodel=load_model(AEmodelLocation)
        CNNmodel=load_model(CNNmodelLocation)
        clientsAEModelList.append(AEmodel)
        clientsCNNModelist.append(CNNmodel)
        lastLowerBound+=clientDataInterval

    for clientID in range(numOfClients):
        x_train_labeled,y_train_labeled,x_train_unlabeled=splitLabel(xClientsList[clientID],yClientsList[clientID])
        x_train_labeled = np.expand_dims(x_train_labeled, axis=2)
        x_train_unlabeled= np.expand_dims(x_train_unlabeled, axis=2)
        xClientsListLabel.append(x_train_labeled)
        xClientsListUnLabel.append(x_train_unlabeled)
        yClientsListLabel.append(y_train_labeled)


    # ------- 3. Update clients' model with intial server's deep-model ----------
    start_time = time.time()
    process = psutil.Process(os.getpid())

    for iterationNo in range(1,numOfIterations+1):
        print("**********************开始第：",iterationNo,"轮全局训练**********************")
        for clientID in range(numOfClients):
            print("=====================开始训练第",clientID,"个子节点====================")
            monitor = Monitor(1,"子节点训练",iterationNo,clientID,process) #delay,stage,iterationNo,clientID,process

            subAEmodel=originautoencoder
            subCNNmodel=originclassificationmodel

            subAEmodel.set_weights(clientsAEModelList[clientID].get_weights())
            subCNNmodel.set_weights(clientsCNNModelist[clientID].get_weights())

            # 编译模型  #先训练encoder
            subAEmodel.compile(loss='mse',optimizer='adam',metrics='mse')
            subAEmodel.fit(xClientsListUnLabel[clientID], xClientsListUnLabel[clientID],  # 输入数据
                           epochs=epochs,
                           shuffle=True,
                           validation_data=(xServer,xServer),
                           verbose=verbose)
            #训练完把encoder的权重拼在cnn前面 接着训练
            subCNNmodel.encoder.set_weights(subAEmodel.encoder.get_weights())

            subCNNmodel.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
            history=subCNNmodel.fit(xClientsListLabel[clientID], yClientsListLabel[clientID],  # 输入数据
                            epochs=epochs,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_data=(xServer,yServer),
                            verbose=verbose)

            monitor.stop()
            y_test_pr =subCNNmodel.predict(xServer, batch_size=300)
            test_acc=accuracy_score(yServer.argmax(-1), y_test_pr.argmax(-1))
            print("Test accuracy : %f" % test_acc)
            performancerdirct['stage']="子节点训练"
            performancerdirct['iterationNo']=iterationNo
            performancerdirct['clientID']=clientID
            performancerdirct['train_acc']=history.history["accuracy"][-1]
            performancerdirct['val_acc']=history.history["val_accuracy"][-1]
            performancerdirct['test_acc']=test_acc
            performancerdirctros.append(performancerdirct.copy())

            clientAEWeight=subAEmodel.get_weights()
            clientCNNWeight=subCNNmodel.get_weights()

            updateServerModel(clientAEWeight,clientCNNWeight)
            subCNNmodel.save("./Models/AECNNmodel/CNN_node_"+str(clientID)+".h5")
            subAEmodel.save("./Models/AECNNmodel/AE_node_"+str(clientID)+".h5")
            firstClientFlag=False


        #Avarage all clients model
        print("=====================子节点训练完毕，开始聚合=====================")
        monitor = Monitor(1,"全局聚合",iterationNo,999999,process)
        #先聚合AE的参数
        for ind in range(len(deepAEModelAggWeights)):
            deepAEModelAggWeights[ind]/=numOfClients
        dw_last=originautoencoder.get_weights()
        for ind in range(len(deepAEModelAggWeights)):
            dw_last[ind]=deepAEModelAggWeights[ind]
        #再聚合CNN的参数
        originautoencoder.set_weights(dw_last)
        originautoencoder.save(AEmodelLocation)
        for ind in range(len(deepCNNModelAggWeights)):
            deepCNNModelAggWeights[ind]/=numOfClients
        dw_last=originclassificationmodel.get_weights()
        for ind in range(len(deepCNNModelAggWeights)):
            dw_last[ind]=deepCNNModelAggWeights[ind]
        originclassificationmodel.set_weights(dw_last)
        originclassificationmodel.save(CNNmodelLocation)
        monitor.stop()
        #Update server's model

        # Servers model is updated, now it can be used again by the clients
        print("=====================聚合完毕，开始下发模型=====================")
        updateClientsModels()
        firstClientFlag=True
        deepCNNModelAggWeights.clear()
        deepAEModelAggWeights.clear()


    #全部训练完 开始验证
    print("================训练全部结束，开始进行验证========================")
    ACC_list=[]

    for clientID in range(numOfClients):
        monitor = Monitor(1,"训练完成进行验证",999999,clientID,process)

        nodemodel=originclassificationmodel
        nodemodel.set_weights(load_model("./Models/AECNNmodel/CNN_node_"+str(clientID)+".h5").get_weights())
        y_test_pr = nodemodel.predict(xServer, batch_size=100)
        endtesttime = datetime.datetime.now()
        acc=accuracy_score(yServer.argmax(-1), y_test_pr.argmax(-1))
        report = classification_report(yServer.argmax(-1), y_test_pr.argmax(-1),target_names= LABELS, output_dict=True)
        monitor.stop()
        print("第",clientID,"个节点的测试accuracy为 : %f" % acc)
        ACC_list.append(acc)
        performancerdirct['stage']="训练后全局验证"
        performancerdirct['clientID']=clientID
        performancerdirct['test_acc']=acc
        performancerdirct['classification_report']=report
        performancerdirctros.append(performancerdirct.copy())

    print("==================================================")
    print("ACC AVG",np.mean(ACC_list))
    with open(monitoring_filename,'a+',newline='')as f:
        f_csv = csv.DictWriter(f,monitorheaders)
        f_csv.writeheader()
        f_csv.writerows(Globalmonitordirctrows)
    with open(performance_filename,'a+',newline='')as f:
        f_csv = csv.DictWriter(f,performanceheaders)
        f_csv.writeheader()
        f_csv.writerows(performancerdirctros)