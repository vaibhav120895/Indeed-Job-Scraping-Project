# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:49:59 2020

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
    next(reader)
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

predictions = []

#Build a counter based on the training dataset
counter = CountVectorizer(stop_words=stopwords.words('english'))
counter.fit(JD_train)



#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(JD_train)#transform the training data
counts_test = counter.transform(JD_test)#transform the testing data

KNN_classifier=KNeighborsClassifier()
LREG_classifier=LogisticRegression(solver='liblinear')
DT_classifier = DecisionTreeClassifier()
RF_classifier = RandomForestClassifier()
#ADB_classifier = AdaBoostClassifier(n_estimators=200)

#ADB_classifier.get_params().keys()

predictors=[('knn',KNN_classifier),('lreg',LREG_classifier),('dt',DT_classifier),('R_Forest', RF_classifier)]

VT=VotingClassifier(predictors, weights=[0.1,0.32,0.23,0.35])

#build the parameter grid
KNN_grid = [{'n_neighbors': [1,3,5,7,9,11,13,15,17], 'weights':['uniform','distance']}]

#build a grid search to find the best parameters
gridsearchKNN = GridSearchCV(KNN_classifier, KNN_grid, cv=5)

#run the grid search
gridsearchKNN.fit(counts_train,JT_train)

#=======================================================================================

#build the parameter grid
DT_grid = [{'max_depth': [3,4,5,6,7,8,9,10,11,12],'criterion':['gini','entropy']}]

#build a grid search to find the best parameters
gridsearchDT  = GridSearchCV(DT_classifier, DT_grid, cv=5)

#run the grid search
gridsearchDT.fit(counts_train,JT_train)

#=======================================================================================

#build the parameter grid
LREG_grid = [ {'C':[0.5,1,1.5,2],'penalty':['l1','l2']}]

#build a grid search to find the best parameters
gridsearchLREG  = GridSearchCV(LREG_classifier, LREG_grid, cv=5)

#run the grid search
gridsearchLREG.fit(counts_train,JT_train)

#=======================================================================================

## RANDOM FOREST GRID
#=======================================================================================

#build the parameter grid
RF_grid = [ { 'n_estimators' : [100, 200,400], 'max_depth' : [5, 10, 15]}]

#build a grid search to find the best parameters
gridsearchRF  = GridSearchCV(RF_classifier, RF_grid, cv=5)

#run the grid search
gridsearchRF.fit(counts_train,JT_train)

#=======================================================================================

# ## Adaboost
# #=======================================================================================

# #build the parameter grid
# ADB_grid = [ {'n_estimators':[100,500,1000],'learning_rate':[.001,0.01,.1]}]

# #build a grid search to find the best parameters
# gridsearchADB  = GridSearchCV(ADB_classifier, LREG_grid, cv=5)

#run the grid search
#gridsearchADB.fit(counts_train,JT_train)

#=======================================================================================

VT.fit(counts_train,JT_train)

#use the VT classifier to predict
predicted=VT.predict(counts_test)
# f=open('MEGAFILE_TEST.csv', encoding="utf8")
# reader2 = csv.reader(f)

df = pd.read_csv("MEGAFILE_TEST.csv", header = None, names = ['JD'])
list1 = df.JD.to_list()
predicted_2 = VT.predict(list1)

# #print the accuracy
print ('Accuracy score for initial test set',accuracy_score(predicted,JT_test))

file = open('Predictions.csv', 'w+', newline ='') 
  
# writing the data into the file 
with file:     
    write = csv.writer(file) 
    write.writerows(predicted_2)

#   #  return 

