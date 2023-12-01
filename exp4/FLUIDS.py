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
labelnum=500  #每个类有标签的数据量 (no use in this code)
latent_dim = 39
#每个模型的参数
verbose, epochs, batch_size =2,5, 64
numOfIterations=50   #全局训练轮数
numOfClients=10  #子节点个数
monitoring_filename= "./result/FLUIDS_"+str(labelnum)+"label_性能监控.csv"
performance_filename= "./result/FLUIDS_"+str(labelnum)+"label_评价指标.csv"
AEmodelLocation="./Models/FLUIDS_AE_"+str(numOfClients)+"_nodes.h5"
ClassficationmodelLocation="./Models/FLUIDS_CF_"+str(numOfClients)+"_nodes.h5"

#模型结构
def createDeepModel(inp_size,n_classes):
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

    #totalmodel= Model(input_e,[mlp(encoder(input_e)),decoder(encoder(input_e))])
    #totalmodel.encoder=encoder
    #totalmodel.test_model=mlp
    return autoencoder,classificationmodel


def splitLabel(x_train,y_train):
    idxs_annot = np.random.choice(x_train.shape[0], labelnum)
    x_train_labeled   = x_train[idxs_annot]
    y_train_labeled   = y_train[idxs_annot]
    return  x_train_labeled,y_train_labeled


#######深度学习相关######

def updateServerModel(clientModelWeight):
    global firstClientFlag
    for ind in range(len(clientModelWeight)):
        if(firstClientFlag==True):
            deepModelAggWeights.append(clientModelWeight[ind])
        else:
            deepModelAggWeights[ind]=(deepModelAggWeights[ind]+clientModelWeight[ind])

def updateClientsModels(originmodel,servermodel):
    clientsModelList.clear()
    for clientID in range(numOfClients):
        m = tensorflow.keras.models.clone_model(originmodel)
        m.set_weights(servermodel.get_weights())
        clientsModelList.append(m)

############统计相关###########
monitorheaders = ['stage','iterationNo','clientID','avg_GPU_mem','avg_GPU_load','avg_Memory_used','avg_cpu_used','used_time(us)']
Globalmonitordirct={'stage':'','iterationNo':0,'clientID':0,'avg_GPU_mem':0,'avg_GPU_load':0,'avg_Memory_used':0,'avg_cpu_used':0,'used_time(us)':0}
Globalmonitordirctrows=[]

performanceheaders=['stage','iterationNo','clientID','train_value','val_value','test_value','classification_report']
performancerdirct={'stage':'','iterationNo':0,'clientID':0,'train_value':0,'val_value':0,'test_value':0,'classification_report':''}
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
    accList, precList, recallList, f1List = [], [], [], []
    deepModelAggWeights=[]
    firstClientFlag=True
    #step1  加载数据集
    dfDS = pd.read_csv(root_path+'./dataset/'+'ISCX_5class_each_normalized_cuttedfloefeature.csv')
    X_full = dfDS.iloc[:, 1:len(dfDS.columns)].values
    Y_full = dfDS["label"].values
    num_classes=len(set(Y_full))
    Y_full = tensorflow.keras.utils.to_categorical(Y_full, num_classes)


    # FOR TEST SPLIT
    xtest, xdata, ytest, ydata = train_test_split(X_full, Y_full, test_size=0.90,random_state=523)
    #xtest 用来进行测试验证的数据 ytest是其对应的标签
    #xdata 是用来训练的数据 ydata 是其对应的标签
    print("xtest",xtest.shape)
    print("xdata",xdata.shape)

    xServer, xClients, yServer, yClients = train_test_split(xdata, ydata, test_size=0.90,random_state=523)
    #xServer 是服务器用来做有监督学习的数据 yServer 是服务器其对应的标签
    #xClients 是其客户端做无监督学习的数据 yClients会被丢弃
    #xServer = np.expand_dims(xServer, axis=2)
    print("xServer",xServer.shape)
    print("xClients",xClients.shape)

    #创建初始模型
    inp_size = xServer.shape[1]
    originautoencoder,originclassificationmodel=createDeepModel(inp_size,num_classes)
    originautoencoder.save(AEmodelLocation)
    originclassificationmodel.save(ClassficationmodelLocation)

# ------- 2. 拆分子节点训练数据 ----------
    xClientsList=[]
    yClientsList=[]
    xClientsListLabel=[]
    yClientsListLabel=[]

    clientsModelList=[]  #存放子节点权重的

    clientDataInterval=len(xClients)//numOfClients
    lastLowerBound=0
    #数据拆分
    for clientID in range(numOfClients):
        xClientsList.append(xClients[lastLowerBound : lastLowerBound+clientDataInterval])
        yClientsList.append(yClients[lastLowerBound : lastLowerBound+clientDataInterval])
        model=load_model(AEmodelLocation)
        clientsModelList.append(model)
        lastLowerBound+=clientDataInterval

    for clientID in range(numOfClients):
        #由于FLUIDS 本地是无监督学习 因此此处不做 有标记数据的筛选 ，全部用于训练
        x_train_labeled,y_train_labeled=xClientsList[clientID],yClientsList[clientID]
        #x_train_labeled = np.expand_dims(x_train_labeled, axis=2)
        xClientsListLabel.append(x_train_labeled)
        yClientsListLabel.append(y_train_labeled)

    # ------- 3. Update clients' model with intial server's deep-model ----------
    start_time = time.time()
    process = psutil.Process(os.getpid())

    for iterationNo in range(1,numOfIterations+1):
        print("**********************开始第：",iterationNo,"轮全局训练**********************")

        if iterationNo==1:
            #如果是第一轮训练 什么也不动
            servermodel=originclassificationmodel
        else:
            monitor = Monitor(1,"全局有监督训练",iterationNo,999999,process) #delay,stage,iterationNo,clientID,process
            servermodel=originclassificationmodel
            servermodel.set_weights(load_model(ClassficationmodelLocation).get_weights())

            #服务端有标记fine-tune
            servermodel.encoder.trainable = False   #将编码器部分设置为不可训练
            servermodel.mlp.trainable = True
            servermodel.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
            history=servermodel.fit(xServer, yServer,  # 输入数据
                        epochs=20,
                        batch_size=batch_size, # 批次大小为 32
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
            print("=====================开始训练第",clientID,"个子节点====================")
            monitor = Monitor(1,"子节点训练",iterationNo,clientID,process) #delay,stage,iterationNo,clientID,process
            #每个子节点获取模型结构
            submodel=originautoencoder
            #每个子节点获取上一轮的权重
            submodel.set_weights(clientsModelList[clientID].get_weights())

            # 本地节点无监督训练
            submodel.encoder.trainable = True
            submodel.compile(loss='mse',optimizer='adam',metrics='mse')
            # 训练模型
            history=submodel.fit(xClientsListLabel[clientID], xClientsListLabel[clientID],  # 输入数据
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

            #此处存储当前节点的权重
            clientWeight=submodel.get_weights()
            #更新全局无监督模型的权重
            updateServerModel(clientWeight)
            submodel.save("./Models/FLUIDS/FLUIDS_node_"+str(clientID)+".h5")
            firstClientFlag=False

        #Avarage all clients model
        print("=====================子节点训练完毕，开始聚合=====================")
        monitor = Monitor(1,"全局聚合",iterationNo,999999,process) #delay,stage,iterationNo,clientID,process
        #FED-AVG 对无监督的全局模型进行聚合
        for ind in range(len(deepModelAggWeights)):
            deepModelAggWeights[ind]/=numOfClients
        #获取autoencoder的模型参数结构
        dw_last=originautoencoder.get_weights()
        for ind in range(len(deepModelAggWeights)):
            dw_last[ind]=deepModelAggWeights[ind]
        monitor.stop()
        #Update server's model
        globalautoencoder=originautoencoder
        globalautoencoder.set_weights(dw_last)
        globalautoencoder.save(AEmodelLocation)
        #此处要把全局无监督模型的encoder权重给到全局有监督模型
        originclassificationmodel.encoder.set_weights(globalautoencoder.encoder.get_weights())
        originclassificationmodel.save(ClassficationmodelLocation)
        # Servers model is updated, now it can be used again by the clients
        print("=====================聚合完毕，开始下发模型=====================")
        updateClientsModels(originautoencoder,globalautoencoder)
        firstClientFlag=True
        deepModelAggWeights.clear()

    #全部训练完 开始验证
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
