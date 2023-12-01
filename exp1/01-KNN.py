import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

labelnum=4000  #Set the quantity of labeled data

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


#Labels for two data sets
# LABELS=['Tiktok','iQiyi Video','Jindong Shopping','Snack Video','QQmusic','QQ','Taobao Shopping','NetEase Cloud Music','Arena Of Valor','WeChat']
LABELS=['chat','email','file','streaming','voip']
root_path=''


#step1  Load data set
dfDS = pd.read_csv('exp1/dataset/ISCX_5class_each_normalized_cuttedfloefeature.csv')
X_full = dfDS.iloc[:, 1:len(dfDS.columns)].values
Y_full = dfDS["label"].values
inp_size =X_full.shape[1]
n_classes=len(set(Y_full))
print("X_full",X_full.shape)
print("n_classes",n_classes)

#step2  Split data set
x_train, x_test, y_train, y_test= train_test_split(X_full, Y_full, test_size = 0.1,random_state=5)
#Pick the labeled data
idxs_annot=getlabelindex(y_train,n_classes,labelnum) #每个类挑选有标签的个数
x_train_labeled  = x_train[idxs_annot]
y_train_labeled  = y_train[idxs_annot]

#step3 create model
model = KNeighborsClassifier()
model.fit(x_train_labeled,y_train_labeled)
y_pred =model.predict(x_test)



#Draw result graph
from sklearn.metrics import confusion_matrix, classification_report
conf_matrix = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(20, 20))
sns.heatmap(conf_matrix, annot=True,  fmt='d', square=True, annot_kws={"fontsize":20})
plt.title('Confusion Matrix')
plt.rcParams['font.sans-serif'] = 'simhei'
tick_marks = np.arange(len(LABELS))
plt.xticks(tick_marks, LABELS,rotation=45,fontsize=20)
plt.yticks(tick_marks, LABELS,rotation=45,fontsize=20)

plt.title("Traffic Classification Confusion Matrix (CNN method)",fontsize=30)
plt.xlabel('Predicted Label',fontsize=25)
plt.ylabel('True Label',fontsize=25)
plt.savefig('Confusion Matrix_KNN_6_{}.png'.format(labelnum),dpi=500)


report = classification_report(y_test, y_pred,target_names= LABELS,digits=4,output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv("classification_report_KNN_6_{}.csv".format(labelnum), index=True)

print(report)
