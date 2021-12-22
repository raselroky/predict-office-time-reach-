#rOkY

################################### KoPaL ######################################

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.preprocessing import LabelEncoder as lec
from sklearn.preprocessing import OrdinalEncoder as oec
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as a_s
from sklearn.metrics import confusion_matrix as c_m
from sklearn.linear_model import LogisticRegression as lor

#load data
path='/home/kalilinux/Desktop/office_time.csv'

df=pd.read_csv(path)

#drop duplicates 
df=df.drop_duplicates()

#encoding
label=lec()

df['catagory']=label.fit_transform(df['catagory'])

#x,y load
x=df[['get-outs-home','reached-bus-stop','stay-bus','catagory']]
y=df['can-reach?']

#train test split
xtrain,xtest,ytrain,ytest=tts(x,y,test_size=.2,random_state=1)

#load or create model object
model=rfc()

model.fit(xtrain,ytrain)

#predict
prd=model.predict(xtest)

#find score and accuracy
sc=model.score(xtest,ytest)
ac=a_s(ytest,prd)

print(sc,ac)


#naive bayes,decission tree,logistic regression,svm,knn etc theke random forest er score besh valo 
