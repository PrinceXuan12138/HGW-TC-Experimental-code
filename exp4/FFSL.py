# -*- coding: utf-8 -*-
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
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten,Convolution1D,Dense, Conv2D, MaxPool2D, UpSampling2D,UpSampling1D,Convolution1D,MaxPooling1D,ZeroPadding1D
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


#labels for the different dataset
#LABELS=['douyin','iqiyi','jindong','kuaishou','qqmusic','qq','taobao','wangyiyunyinyue','wangzherongyao','weixin']
LABELS=['chat','email','file','streaming','voip']
root_path=''
latent_dim = 39
verbose, epochs, batch_size =1,5, 64  # Parameters for local training
numOfIterations=50    #global epochs
numOfClients=10  #num of sub nodes
monitoring_filename= "./result/FFSL_label_性能监控.csv"  #Path of the performance monitoring record file
performance_filename= "./result/FFSL_label_评价指标.csv"  #Path of the performance  index record file
AEmodelLocation="./Models/FFSL_AE_"+str(numOfClients)+"_nodes.h5"  #Path of the server model
ClassficationmodelLocation="./Models/FFSL_CF_"+str(numOfClients)+"_nodes.h5" #Path of the server model

########Define model##########
def createDeepModel(inp_size,n_classes):
    #encoder
    input_shape=(inp_size,1)
    input_e = Input(shape=input_shape)
    x =Convolution1D(64,3,padding="same",activation="relu",name='conv_1')(input_e)
    x =Convolution1D(64,3,padding="same",activation="relu",name='conv_2')(x)
    x=MaxPooling1D(name='maxpool_3')(x)
    x=Convolution1D(32,3,padding="same",activation="relu",name='conv_4')(x)
    x=MaxPooling1D(name='maxpool_5')(x)
    s_shape = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(latent_dim,activation="relu",name='FC6')(x)
    encoder = Model(input_e,latent,name="encoder")
    #decoder
    input_d = Input(shape=(latent_dim,))
    x = Dense(s_shape[1] * s_shape[2])(input_d)
    x = Reshape((s_shape[1], s_shape[2]))(x)
    x = UpSampling1D(name='upsample_2')(x)
    x = Convolution1D(32, 3, padding="same", activation="relu", name='deconv_1')(x)
    x = UpSampling1D(name='upsample_4')(x)
    x = Convolution1D(32, 3, padding="same", activation="relu", name='deconv_3')(x)
    x = Convolution1D(64, 3, padding="same", activation="relu", name='deconv_5')(x)
    x = ZeroPadding1D(padding=(1, 1), name='zeropad_6')(x)
    output_d = Convolution1D(1, 3, padding="same", activation="sigmoid", name='deconv_6')(x)
    decoder = Model(input_d, output_d, name="decoder")

    # decoder.summary()
    #mlp
    input_c = Input(shape=(latent_dim,))
    x = Dense(np.prod(s_shape[1:]))(input_c)
    x = Dense(128,activation="relu",name='FC7')(x)
    x = Dense(64,activation="relu",name='FC8')(x)
    x = Dense(n_classes,activation="softmax")(x)
    mlp = Model(input_c,x,name="mlp")

    autoencoder=Model(input_e,decoder(encoder(input_e)))
    autoencoder.encoder=encoder
    classificationmodel=Model(input_e,mlp(encoder(input_e)))
    classificationmodel.encoder=encoder
    classificationmodel.mlp=mlp

    return autoencoder,classificationmodel




#######Deep learning related code######

def updateServerModel(clientModelWeight):
    global firstClientFlag
    for ind in range(len(clientModelWeight)):
        if(firstClientFlag==True):
            deepModelAggWeights.append(clientModelWeight[ind])
        else:
            deepModelAggWeights[ind]=(deepModelAggWeights[ind]+clientModelWeight[ind])

def updateClientsModels(originmodel,servermodel):
    clientsModelList.clear()
    K.clear_session()
    for clientID in range(numOfClients):
        m = tensorflow.keras.models.clone_model(originmodel)
        m.set_weights(servermodel.get_weights())
        clientsModelList.append(m)

############Statistical parameter###########
monitorheaders = ['stage','iterationNo','clientID','avg_GPU_mem','avg_GPU_load','avg_Memory_used','avg_cpu_used','used_time(us)']
Globalmonitordirct={'stage':'','iterationNo':0,'clientID':0,'avg_GPU_mem':0,'avg_GPU_load':0,'avg_Memory_used':0,'avg_cpu_used':0,'used_time(us)':0}
Globalmonitordirctrows=[]

performanceheaders=['stage','iterationNo','clientID','train_value','val_value','test_value','classification_report']
performancerdirct={'stage':'','iterationNo':0,'clientID':0,'train_value':0,'val_value':0,'test_value':0,'classification_report':''}
performancerdirctros=[]


#Code performance monitoring
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
    accList, precList, recallList, f1List = [], [], [], []
    deepModelAggWeights=[]
    firstClientFlag=True
    #step1  load data
    dfDS = pd.read_csv(root_path+'./dataset/'+'ISCX_5class_each_normalized_cuttedfloefeature.csv')
    X_full = dfDS.iloc[:, 1:len(dfDS.columns)].values
    Y_full = dfDS["label"].values
    num_classes=len(set(Y_full))
    Y_full = tensorflow.keras.utils.to_categorical(Y_full, num_classes)


    # split data
    xtest, xdata, ytest, ydata = train_test_split(X_full, Y_full, test_size=0.90,random_state=523)
    # xtest is used for test verification ytest is its corresponding label.
    # xdata is  used for training ydata is its corresponding label
    print("xtest",xtest.shape)
    print("xdata",xdata.shape)

    xServer, xClients, yServer, yClients = train_test_split(xdata, ydata, test_size=0.90,random_state=523)

    # xServer is the data used by the server for supervised learning. YServer is the corresponding label of the server.
    # xClients is the data that its client does unsupervised learning yClients will be discarded

    print("xServer",xServer.shape)
    print("xClients",xClients.shape)

    #create initial model
    inp_size = xServer.shape[1]
    originautoencoder,originclassificationmodel=createDeepModel(inp_size,num_classes)
    originautoencoder.save(AEmodelLocation)
    originclassificationmodel.save(ClassficationmodelLocation)

    # ------- 2. The training data is split according to the number of sub nodes ----------
    xClientsList=[]
    yClientsList=[]
    xClientsListLabel=[]
    yClientsListLabel=[]

    clientsModelList=[]   #Store the model of the sub node

    clientDataInterval=len(xClients)//numOfClients
    lastLowerBound=0
    #Split the data by number of sub nodes
    for clientID in range(numOfClients):
        xClientsList.append(xClients[lastLowerBound : lastLowerBound+clientDataInterval])
        yClientsList.append(yClients[lastLowerBound : lastLowerBound+clientDataInterval])
        model=load_model(AEmodelLocation)
        clientsModelList.append(model)
        lastLowerBound+=clientDataInterval

    for clientID in range(numOfClients):
        # because FFSL is locally unsupervised learning, there is no filtering of tagged data here, all for training
        x_train_labeled,y_train_labeled=xClientsList[clientID],yClientsList[clientID]

        xClientsListLabel.append(x_train_labeled)
        yClientsListLabel.append(y_train_labeled)

    # ------- 3. train process ----------
    start_time = time.time()
    process = psutil.Process(os.getpid())

    for iterationNo in range(1,numOfIterations+1):
        print("**********************开始第：",iterationNo,"轮全局训练**********************")
        # each global epoch
        if iterationNo==1:
            # if it's the first round of training, nothing moves.
            servermodel=originclassificationmodel
        else:
            monitor = Monitor(1,"全局有监督训练",iterationNo,999999,process) #delay,stage,iterationNo,clientID,process
            servermodel=originclassificationmodel
            servermodel.set_weights(load_model(ClassficationmodelLocation).get_weights())

            #The server uses labeled data fine-tune
            servermodel.encoder.trainable = False  # Set encoder to untrainable
            servermodel.mlp.trainable = True
            servermodel.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
            history=servermodel.fit(xServer, yServer,
                                    epochs=20,
                                    batch_size=batch_size,
                                    validation_data=(xtest,ytest),
                                    verbose=verbose)
            servermodel.save(ClassficationmodelLocation, save_format='tf')
            monitor.stop()
            performancerdirct['stage']="子节点训练"
            performancerdirct['iterationNo']=iterationNo
            performancerdirct['clientID']=999999
            performancerdirct['train_value']=history.history["accuracy"][-1]
            performancerdirct['val_value']=history.history["val_accuracy"][-1]
            performancerdirct['test_value']=history.history["val_accuracy"][-1]
            performancerdirctros.append(performancerdirct.copy())

        for clientID in range(numOfClients):
            #each local epoch
            print("=====================开始训练第",clientID,"个子节点====================")
            monitor = Monitor(1,"子节点训练",iterationNo,clientID,process) #delay,stage,iterationNo,clientID,process
            # Each sub node gets the model structure
            submodel=originautoencoder
            #Each sub node gets the weight of the previous round
            submodel.set_weights(clientsModelList[clientID].get_weights())

            # sub node unsupervised local training
            submodel.encoder.trainable = True
            submodel.compile(loss='mse',optimizer='adam',metrics='mse')
            history=submodel.fit(xClientsListLabel[clientID], xClientsListLabel[clientID],
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(xtest,xtest)
                                 ,verbose=verbose)
            monitor.stop()
            performancerdirct['stage']="子节点训练"
            performancerdirct['iterationNo']=iterationNo
            performancerdirct['clientID']=clientID
            performancerdirct['train_value']=history.history["mse"][-1]
            performancerdirct['val_value']=history.history["val_mse"][-1]
            performancerdirct['test_value']=history.history["val_mse"][-1]
            performancerdirctros.append(performancerdirct.copy())

            # The weight of the current node is stored here
            clientWeight=submodel.get_weights()
            # Update the weights of the global unsupervised model
            #Add the weights of the model for each sub node
            updateServerModel(clientWeight)
            submodel.save("./Models/FFSL/FFSL_node_"+str(clientID)+".h5")
            firstClientFlag=False

        #Avarage all clients model
        print("=====================子节点训练完毕，开始聚合=====================")
        monitor = Monitor(1,"全局聚合",iterationNo,999999,process) #delay,stage,iterationNo,clientID,process

        #Average the weights that are accumulated in the for loop----- (FedAVG)
        #According to the paper, only unsupervised models are aggregated here

        for ind in range(len(deepModelAggWeights)):
            deepModelAggWeights[ind]/=numOfClients
        #The weight of the resulting aggregate model is used as the weight of the new initial model
        dw_last=originautoencoder.get_weights()
        for ind in range(len(deepModelAggWeights)):
            dw_last[ind]=deepModelAggWeights[ind]
        monitor.stop()
        #Update server's model
        globalautoencoder=originautoencoder
        globalautoencoder.set_weights(dw_last)
        globalautoencoder.save(AEmodelLocation)

        # Here the encoder of the globally unsupervised model is weighted to the globally supervised model
        originclassificationmodel.encoder.set_weights(globalautoencoder.encoder.get_weights())
        originclassificationmodel.save(ClassficationmodelLocation)
        # Servers model is updated, now it can be used again by the clients
        print("=====================聚合完毕，开始下发模型=====================")
        updateClientsModels(originautoencoder,globalautoencoder)
        firstClientFlag=True
        deepModelAggWeights.clear()

    #Start verification after all training
    print("================训练全部结束，开始进行验证========================")
    ACC_list=[]
    nodemodel=originclassificationmodel
    nodemodel.set_weights(load_model(ClassficationmodelLocation).get_weights())
    monitor = Monitor(1,"训练完成进行验证",999999,999999,process)
    y_test_pr = nodemodel.mlp.predict(nodemodel.encoder(xtest), batch_size=100)
    endtesttime = datetime.datetime.now()
    acc=accuracy_score(ytest.argmax(-1), y_test_pr.argmax(-1))
    report = classification_report(ytest.argmax(-1), y_test_pr.argmax(-1),target_names= LABELS, output_dict=True)
    monitor.stop()
    print("最终的测试accuracy为 : %f" % acc)
    ACC_list.append(acc)
    performancerdirct['stage']="训练后全局验证"
    performancerdirct['clientID']=99999
    performancerdirct['test_value']=acc
    performancerdirct['classification_report']=report
    performancerdirctros.append(performancerdirct.copy())

    # for clientID in range(numOfClients):
    #     monitor = Monitor(1,"训练完成进行验证",999999,clientID,process)
    #     y_test_pr = nodemodel.test_model.predict(nodemodel.encoder(xtest), batch_size=100)
    #     endtesttime = datetime.datetime.now()
    #     acc=accuracy_score(ytest.argmax(-1), y_test_pr.argmax(-1))
    #     report = classification_report(ytest.argmax(-1), y_test_pr.argmax(-1),target_names= LABELS, output_dict=True)
    #     monitor.stop()
    #     print("第",clientID,"个节点的测试accuracy为 : %f" % acc)
    #     ACC_list.append(acc)
    #     performancerdirct['stage']="训练后全局验证"
    #     performancerdirct['clientID']=clientID
    #     performancerdirct['test_acc']=acc
    #     performancerdirct['classification_report']=report
    #     performancerdirctros.append(performancerdirct.copy())

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
