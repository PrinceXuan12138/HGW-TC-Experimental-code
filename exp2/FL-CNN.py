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
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Conv1D, Dense, Flatten
from tensorflow.keras.models import load_model
import random
from sklearn.model_selection import train_test_split
from threading import Thread
import GPUtil
import csv
from sklearn.metrics import classification_report

#labels for the different dataset
#LABELS=['douyin','iqiyi','jindong','kuaishou','qqmusic','qq','taobao','wangyiyunyinyue','wangzherongyao','weixin']
LABELS=['chat','email','file','streaming','voip']
root_path=''
labelnum=500  # #The amount of labeled data (per class)

verbose, epochs, batch_size =0,5, 256  # Parameters for local training
numOfIterations=50    #global epochs
numOfClients=10   #num of sub nodes
monitoring_filename= "./result/联邦学习CNN_"+str(labelnum)+"label_性能监控.csv" #Path of the performance monitoring record file
performance_filename= "./result/联邦学习CNN_"+str(labelnum)+"label_评价指标.csv"  #Path of the performance  index record file
modelLocation="./Models/FL_CNN_"+str(numOfClients)+"_nodes.h5"   #Path of the server model

########Define model##########
def createDeepModel():

    model=Sequential()
    model.add(Conv1D(filters=10, kernel_size=3, activation='relu',input_shape = (inp_size, 1)))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Conv1D(filters=5, kernel_size=3, activation='relu'))
    model.add(Dropout(0.05))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def splitLabel(x_train,y_train):
    idxs_annot = range(x_train.shape[0])
    random.seed(0)
    idxs_annot = np.random.choice(x_train.shape[0], labelnum)
    x_train_labeled   = x_train[idxs_annot]
    y_train_labeled   = y_train[idxs_annot]
    return  x_train_labeled,y_train_labeled

#######Deep learning related code######
accList, precList, recallList, f1List = [], [], [], []
deepModelAggWeights=[]
firstClientFlag=True

def trainInServer(model,x_train_labeled,y_train_labeled):
    model.fit(x_train_labeled, y_train_labeled, epochs=epochs, batch_size=batch_size,validation_split= 0.3,verbose=verbose)
def updateServerModel(clientModel, clientModelWeight):
    global firstClientFlag
    for ind in range(len(clientModelWeight)):
        if(firstClientFlag==True):
            deepModelAggWeights.append(clientModelWeight[ind])
        else:
            deepModelAggWeights[ind]=(deepModelAggWeights[ind]+clientModelWeight[ind])
def updateClientsModels():
    global clientsModelList
    global deepModel
    clientsModelList.clear()
    for clientID in range(numOfClients):
        m = tensorflow.keras.models.clone_model(deepModel)
        m.set_weights(deepModel.get_weights())
        clientsModelList.append(m)
        # m = Globalmodel
        # m.set_weights(Globalmodel.get_weights())
        # clientsModelList.append(m)

############Statistical parameter###########
monitorheaders = ['stage','iterationNo','clientID','avg_GPU_mem','avg_GPU_load','avg_Memory_used','avg_cpu_used','used_time(us)']
Globalmonitordirct={'stage':'','iterationNo':0,'clientID':0,'avg_GPU_mem':0,'avg_GPU_load':0,'avg_Memory_used':0,'avg_cpu_used':0,'used_time(us)':0}
Globalmonitordirctrows=[]

performanceheaders=['stage','iterationNo','clientID','train_acc','val_acc','test_acc','classification_report']
performancerdirct={'stage':'','iterationNo':0,'clientID':0,'train_acc':0,'val_acc':0,'test_acc':0,'classification_report':''}
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
        Globalmonitordirct['used_time(us)']=used_time

        Globalmonitordirctrows.append(Globalmonitordirct.copy())


    def stop(self):
        self.stopped = True
        return self.gpu_mem_list,self.gpu_load_list


if __name__ == '__main__':
    #step1  load data
    dfDS = pd.read_csv(root_path+'./dataset/'+'ISCX_5class_each_normalized_cuttedfloefeature.csv')
    X_full = dfDS.iloc[:, 1:len(dfDS.columns)].values
    Y_full = dfDS["label"].values
    num_classes=len(set(Y_full))
    Y_full = keras.utils.to_categorical(Y_full, num_classes)

    # split data
    xServer, xClients, yServer, yClients = train_test_split(X_full, Y_full, test_size=0.90,random_state=523)
    print("yServer",yServer.shape)
    print("yClients",yClients.shape)
    xServer = np.expand_dims(xServer, axis=2)

    #create initial model
    inp_size = xServer.shape[1]
    deepModel=createDeepModel()
    deepModel.save(modelLocation)

    # -----  1. train initial model in at central server (not use in the paper)- -----
    # x_train_labeled,y_train_labeled=splitLabel(xServer,yServer)
    # x_train_labeled = np.expand_dims(x_train_labeled, axis=2)
    # servermodel=initialmodel
    # trainInServer(servermodel,x_train_labeled,y_train_labeled)

    # -------  2. The training data is split according to the number of sub nodes ----------
    xClientsList=[]
    yClientsList=[]
    xClientsListLabel=[]
    yClientsListLabel=[]

    clientsModelList=[]
    clientDataInterval=len(xClients)//numOfClients
    lastLowerBound=0
    #Split the data by number of sub nodes
    for clientID in range(numOfClients):
        xClientsList.append(xClients[lastLowerBound : lastLowerBound+clientDataInterval])
        yClientsList.append(yClients[lastLowerBound : lastLowerBound+clientDataInterval])
        model=load_model(modelLocation)
        clientsModelList.append(model)
        lastLowerBound+=clientDataInterval
    #Split the  labelled data by number of sub nodes
    for clientID in range(numOfClients):
        x_train_labeled,y_train_labeled=splitLabel(xClientsList[clientID],yClientsList[clientID])
        x_train_labeled = np.expand_dims(x_train_labeled, axis=2)
        xClientsListLabel.append(x_train_labeled)
        yClientsListLabel.append(y_train_labeled)

    # ------- 3. train process ----------
    start_time = time.time()
    process = psutil.Process(os.getpid())

    for iterationNo in range(1,numOfIterations+1):
        # each global epoch
        print("**********************开始第：",iterationNo,"轮全局训练**********************")
        for clientID in range(numOfClients):
            #each local epoch
            print("=====================开始训练第",clientID,"个子节点====================")
            monitor = Monitor(1,"子节点训练",iterationNo,clientID,process) #delay,stage,iterationNo,clientID,process
            clientsModelList[clientID].compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
            history=clientsModelList[clientID].fit(xClientsListLabel[clientID], yClientsListLabel[clientID], epochs=epochs, batch_size=batch_size,validation_split= 0.3,verbose=verbose)
            monitor.stop()
            # scores = clientsModelList[clientID].evaluate(xServer, yServer, verbose=verbose)
            # print("CNN TEST Accuracy: ", scores[1])
            y_test_pr = clientsModelList[clientID].predict(xServer, batch_size=100)
            test_acc=accuracy_score(yServer.argmax(-1), y_test_pr.argmax(-1))
            print("Test accuracy : %f" % test_acc)

            performancerdirct['stage']="子节点训练"
            performancerdirct['iterationNo']=iterationNo
            performancerdirct['clientID']=clientID
            performancerdirct['train_acc']=history.history["accuracy"][-1]
            performancerdirct['val_acc']=history.history["val_accuracy"][-1]
            performancerdirct['test_acc']=test_acc
            performancerdirctros.append(performancerdirct.copy())

            clientWeight=clientsModelList[clientID].get_weights()
            #Add the weights of the model for each sub node
            updateServerModel(clientsModelList[clientID], clientWeight)
            clientsModelList[clientID].save("./Models/CNNmodel/CNN_node_"+str(clientID)+".h5")
            firstClientFlag=False

        #Avarage all clients model
        print("=====================子节点训练完毕，开始聚合=====================")
        monitor = Monitor(1,"全局聚合",iterationNo,999999,process) #delay,stage,iterationNo,clientID,process
        for ind in range(len(deepModelAggWeights)):
            deepModelAggWeights[ind]/=numOfClients
        dw_last=deepModel.get_weights()
        for ind in range(len(deepModelAggWeights)):
            dw_last[ind]=deepModelAggWeights[ind]
        monitor.stop()
        #Update server's model
        deepModel.set_weights(dw_last)
        # Servers model is updated, now it can be used again by the clients
        print("=====================聚合完毕，开始下发模型=====================")
        updateClientsModels()
        firstClientFlag=True
        deepModelAggWeights.clear()

    #Start verification after all training
    print("================训练全部结束，开始进行验证========================")
    ACC_list=[]
    for clientID in range(numOfClients):
        monitor = Monitor(1,"训练完成进行验证",999999,clientID,process)
        starttesttime=datetime.datetime.now()
        nodemodel=load_model("./Models/CNNmodel/CNN_node_"+str(clientID)+".h5")
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
    #Start verification after all training
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
    deepModel.summary()