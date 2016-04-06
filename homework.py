#coding:utf-8
'''
nie
2016/4/6
assignments for data mining

using Decision tree、svm、logistic regression to modeling the data and comparison results
'''

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import random

def svc(traindata,trainlabel,testdata,testlabel,show_accuracy=False):
    #print("Start training SVM classifier...")
    svcClf = SVC(C=1.0,kernel="rbf",cache_size=3000)
    svcClf.fit(traindata,trainlabel)
    
    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    if show_accuracy:
        print("SVM Accuracy:",accuracy)
        print(classification_report(testlabel,pred_testlabel))
    return accuracy,precision_recall_fscore_support(testlabel,pred_testlabel)

def rf(traindata,trainlabel,testdata,testlabel,show_accuracy=False):
    #print("Start training Random Forest classifier...")
    rfClf = RandomForestClassifier(n_estimators=400,criterion='gini')
    rfClf.fit(traindata,trainlabel)
    
    pred_testlabel = rfClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    if show_accuracy:
        print("Random Forest Accuracy:",accuracy)
        print(classification_report(testlabel,pred_testlabel))
    return accuracy,precision_recall_fscore_support(testlabel,pred_testlabel)

def nb(traindata,trainlabel,testdata,testlabel,show_accuracy=False):
    #print("Start training naive bayes classifier...")
    nbclf=GaussianNB()
    nbclf.fit(traindata,trainlabel)
    pred_testlabel=nbclf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    if show_accuracy:
        print("navie bayes Accuracy:",accuracy)
        print(classification_report(testlabel,pred_testlabel))
    return accuracy,precision_recall_fscore_support(testlabel,pred_testlabel)

def lr(traindata,trainlabel,testdata,testlabel,show_accuracy=False):
    #print("Start training logistic regression classifier...")
    lr=LogisticRegression()
    lr.fit(traindata,trainlabel)
    pred_testlabel=lr.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    if show_accuracy:
        print("logistic regression Accuracy:",accuracy)
        print(classification_report(testlabel,pred_testlabel))
    return accuracy,precision_recall_fscore_support(testlabel,pred_testlabel)

def tr(traindata,trainlabel,testdata,testlabel,show_accuracy=False):
    #print("Start training decision tree classifier ...")
    tr=DecisionTreeClassifier()
    tr.fit(traindata,trainlabel)
    pred_testlabel=tr.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    if show_accuracy:
        print("decision tree Accuracy:",accuracy)
        print(classification_report(testlabel,pred_testlabel))
    return accuracy,precision_recall_fscore_support(testlabel,pred_testlabel)

if __name__=='__main__':
    rawdata=np.loadtxt('data/sample.csv',delimiter=',')
    label=rawdata[:,-1]
    label=np.asarray(label,dtype='int')
    data=rawdata[:,:-1]
    print("样本数目与样本维数: "+str(data.shape)+"\n")

    K=10  # k-fold
    train_data=[]
    test_data=[]
    train_label=[]
    test_label=[]

    kf = KFold(len(data), n_folds=K,shuffle=True) # K-fold
    for train_index, test_index in kf:
        train_data.append(data[train_index]);test_data.append(data[test_index])
        train_label.append(label[train_index]);test_label.append(label[test_index])

    

    Acc=np.empty((K),dtype="float32")
    precision=[]
    recall=[]
    f1_score=[]
    print("Start training SVM classifier...")
    for i in range(K):
        acc,prf=svc(train_data[i],train_label[i],test_data[i],test_label[i],show_accuracy=False)
        Acc[i]=acc
        precision.append(prf[0]);
        recall.append(prf[1])
        f1_score.append(prf[2])
    print("average accuracy  of SVM classifier: %f"%Acc.mean())
    print("average precision of SVM classifier: "+str(np.mean(precision,axis=0)))
    print("average recall    of SVM classifier: "+str(np.mean(recall,axis=0)))
    print("average f1_score  of SVM classifier: "+str(np.mean(f1_score,axis=0)))

    Acc=np.empty((K),dtype="float32")
    precision=[]
    recall=[]
    f1_score=[]
    print("\nStart training Random Forest classifier...")
    for i in range(K):
        acc,prf=rf(train_data[i],train_label[i],test_data[i],test_label[i],show_accuracy=False)
        Acc[i]=acc
        precision.append(prf[0]);
        recall.append(prf[1])
        f1_score.append(prf[2])
    print("average accuracy  of Random Forest classifier: %f"%Acc.mean())
    print("average precision of Random Forest classifier: "+str(np.mean(precision,axis=0)))
    print("average recall    of Random Forest classifier: "+str(np.mean(recall,axis=0)))
    print("average f1_score  of Random Forest classifier: "+str(np.mean(f1_score,axis=0)))

    Acc=np.empty((K),dtype="float32")
    precision=[]
    recall=[]
    f1_score=[]
    print("\nStart training naive bayes classifier...")
    for i in range(K):
        acc,prf=nb(train_data[i],train_label[i],test_data[i],test_label[i],show_accuracy=False)
        Acc[i]=acc
        precision.append(prf[0]);
        recall.append(prf[1])
        f1_score.append(prf[2])
    print("average accuracy  of naive bayes classifier: %f"%Acc.mean())
    print("average precision of naive bayes classifier: "+str(np.mean(precision,axis=0)))
    print("average recall    of naive bayes classifier: "+str(np.mean(recall,axis=0)))
    print("average f1_score  of naive bayes classifier: "+str(np.mean(f1_score,axis=0)))

    Acc=np.empty((K),dtype="float32")
    precision=[]
    recall=[]
    f1_score=[]
    print("\nStart training logistic regression classifier...")
    for i in range(K):
        acc,prf=lr(train_data[i],train_label[i],test_data[i],test_label[i],show_accuracy=False)
        Acc[i]=acc
        precision.append(prf[0]);
        recall.append(prf[1])
        f1_score.append(prf[2])
    print("average accuracy  of logistic regression classifier: %f"%Acc.mean())
    print("average precision of logistic regression classifier: "+str(np.mean(precision,axis=0)))
    print("average recall    of logistic regression classifier: "+str(np.mean(recall,axis=0)))
    print("average f1_score  of logistic regression classifier: "+str(np.mean(f1_score,axis=0)))

    Acc=np.empty((K),dtype="float32")
    precision=[]
    recall=[]
    f1_score=[]
    print("\nStart training decision tree classifier...")
    for i in range(K):
        acc,prf=lr(train_data[i],train_label[i],test_data[i],test_label[i],show_accuracy=False)
        Acc[i]=acc
        precision.append(prf[0]);
        recall.append(prf[1])
        f1_score.append(prf[2])
    print("average accuracy  of decision tree classifier: %f"%Acc.mean())
    print("average precision of decision tree classifier: "+str(np.mean(precision,axis=0)))
    print("average recall    of decision tree classifier: "+str(np.mean(recall,axis=0)))
    print("average f1_score  of decision tree classifier: "+str(np.mean(f1_score,axis=0)))
    
