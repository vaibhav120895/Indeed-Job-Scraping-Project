# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:15:30 2020

@author: Dell
"""

import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import re

from sklearn.ensemble import AdaBoostClassifier


#def GetResult(inputfile):
data = pd.read_csv("DATASET.csv")

train, test = train_test_split(data, test_size=0.30, random_state = 45)
train.info()
test.info()


train.to_csv("DATASET_Train.csv", index = False)
#print(train.head(5))

test.to_csv( "DATASET_Test.csv", index = False)



#read the reviews and their polarities from a given file
def loadData(fname):
    Job_Description=[]
    Job_Title=[]
    f=open(fname, encoding="utf8")
    reader = csv.reader(f)
    for line,JD in reader:
        Text = line
        Title = JD
        try:
            Text=re.sub('data sci[a-z]+', ' ', Text, re.I)
            Text=re.sub('data eng[a-z]+', ' ', Text, re.I)
            Text=re.sub('software eng[a-z]+', ' ', Text, re.I)
            Text=re.sub('  ', ' ', Text, re.I)
            Job_Description.append(Text.lower())    
            Job_Title.append(Title)
    
        except:
            Job_Description.append(Text.lower())    
            Job_Title.append(Title)   
    f.close()
    
    return Job_Description, Job_Title

# def deletecolumn(fname):
#     f= open()

#loadData('DATASET.csv')



#Train Tsr= train_test_split(Job_Description, Job_Title, test_size=0.25)

JD_train, JT_train= loadData('DATASET_Train.csv')
JD_test, JT_test= loadData('DATASET_Test.csv')


counter = CountVectorizer()
counter.fit(JD_train)


#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(JD_train)#transform the training data
counts_test = counter.transform(JD_test)#transform the testing data

print(counts_test)
#train classifier
DT = DecisionTreeClassifier()
ADB = AdaBoostClassifier(n_estimators=200)
LogReg = LogisticRegression(solver='liblinear')
KNN =KNeighborsClassifier()
RF =  RandomForestClassifier()

#train all classifier on the same datasets
DT.fit(counts_train,JT_train)
ADB.fit(counts_train, JT_train)
LogReg.fit(counts_train, JT_train)
KNN.fit(counts_train, JT_train)
RF.fit(counts_train, JT_train)

# #use hard voting to predict (majority voting)
pred=DT.predict(counts_test)
pred1=ADB.predict(counts_test)
pred2=LogReg.predict(counts_test)
pred3 = KNN.predict(counts_test)
pred4 = RF.predict(counts_test)

# #print accuracy
print ("Decision Tree",accuracy_score(pred,JT_test))
print ("AdaBoost" ,accuracy_score(pred1,JT_test))
print ("Logistic",accuracy_score(pred2,JT_test))
print ("KNN",accuracy_score(pred3,JT_test))
print ("RF",accuracy_score(pred2,JT_test))
