# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 23:44:35 2021

@author: HP
"""
""" öncelikle şunu belirtmek isterim ki ilk defa bir veri setini inceleyeceğim için iyice inceleyip anlayarak kendim de ekleme ve 
düzenlemeler yaparak https://www.kaggle.com/petraneumann/data-analysis-titanic-data-python buradan ve youtube dan yardım aldım yazarken """
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv') #test verimiz
y_test = pd.read_csv('gender_submission.csv')


#verimizi okuyalım

print(train.tail())
print(train.columns.tolist()) #sütün isimlerine bakalım
print(train.dtypes)
print(test.columns.tolist())


print(train.isnull().sum()[train.isnull().sum() > 0]) 

"""age,cabin ve embarked sütunlarında null değerler ve bunları halletmeliyiz
bunun için de her bir sütunun ayrı ayrı inceleyelim"""

print(train['Embarked'].value_counts())
#S değeri daha fazla olduğu için nan değerleri de S ile dolduralım

train["Embarked"]=train["Embarked"].fillna("S")
print(train["Embarked"]) #başarılı

"""cabin sütünuna bakalım daha önce 687 tane değerin nan olduğunu görmüşütük 
zatem 891 tane sütün var bu yüzden cabin sütünunu silerim"""

train = train.drop(['Cabin','Ticket','Name','PassengerId'],axis=1)


train['Embarked'].replace('S', 0,inplace=True)
train['Embarked'].replace('C', 1,inplace=True)
train['Embarked'].replace('Q', 2,inplace=True)

#şimdi de age sütnundaki verileri dolduralım
train["Age"]=train["Age"].fillna(train["Age"].median())
print(train["Age"].isnull().sum()) #başarılı

                 
train['Sex'].replace('female', 0,inplace=True)
train['Sex'].replace('male', 1,inplace=True)

test = test.drop(['Cabin','Ticket','Name','PassengerId'],axis=1)

test["Age"]=test["Age"].fillna(test["Age"].median())

                 
print(test["Fare"].value_counts())
# 7.7500 değeri en fazla tekrar ettiği için nan değerleri 7.7500 ile doldururum
test["Fare"]=test["Fare"].fillna(7.7500)
   
test['Sex'].replace('female', 0,inplace=True)
test['Sex'].replace('male', 1,inplace=True)


test['Embarked'].replace('S', 0,inplace=True)
test['Embarked'].replace('C', 1,inplace=True)
test['Embarked'].replace('Q', 2,inplace=True)

x_test =test
x_train = train.drop("Survived", axis=1)
y_train = train["Survived"]
log = LogisticRegression(solver='liblinear')
log.fit(x_train,y_train)
y_pred = log.predict(x_test)
y_test = np.array(y_test["Survived"])

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))