#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 20:26:25 2020

@author: chongshanxie
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, \
    KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets
from sklearn import ensemble
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss, mean_squared_error,SCORERS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV, StratifiedKFold


#### The next part is data prepossing part

data = pd.read_csv('/Users/chongshanxie/Desktop/2020Fall/MSBD5001/kaggle/train.csv',dtype=object)
data_test=pd.read_csv('/Users/chongshanxie/Desktop/2020Fall/MSBD5001/kaggle/test.csv',dtype=object)

data.head()
data['day'], data['month'] , data['year'] = data['date'].str.split('/').str
data_test['day'], data_test['month'] , data_test['year'] = data_test['date'].str.split('/').str

data['year'],data['time'] = data['year'].str.split(' ').str
data_test['year'],data_test['time'] = data_test['year'].str.split(' ').str

data.head()
data['speed']=data['speed'].astype('float')

num=[31,28,31,30,31,30,31,31,30,31,30,31]
num_sum=np.zeros(12)
num_sum[0]=num[0]
for i in range(1,12):
  num_sum[i]=num_sum[i-1]+num[i]


print(num_sum)

train= data[['day','month','year']]
train
train['hour'],train['period']=data['time'].str.split(':0').str
train= train[['day','month','year','hour','period']]
train.head()

test=  data_test[['day','month','year']]
test['hour'],test['period']=data_test['time'].str.split(':0').str
test= test[['day','month','year','hour','period']]
test.head()

for i in range(len(train)):
  month=int(train['month'][i])
  day=int(train['day'][i])
  year=int(train['year'][i])-2017
  x=(num_sum[month-1]+day+year)%7+1
  train['period'][i]=x
  
for i in range(len(test)):
  month=int(test['month'][i])
  day=int(test['day'][i])
  x=(num_sum[month-1]+day+1)%7+1
  test['period'][i]=x

train.head()
test.head()

y=data['speed']
y.head


train.dtypes

train['year']=train['year'].astype('int')
train['day']=train['day'].astype('int')
train['month']=train['month'].astype('int')
train['hour']=train['hour'].astype('int')
train['period']=train['period'].astype('float').astype('int')
test['day']=test['day'].astype('int')
test['year']=test['year'].astype('int')
test['month']=test['month'].astype('int')
test['hour']=test['hour'].astype('int')
test['period']=test['period'].astype('float').astype('int')

#### The next part is feature engineering part, in this part, we will create feature for the model

#day,month,hour,period
train['day_cycle_1']=train['day'].apply(lambda x: np.sin((x - 1) / 31 * np.pi) * np.pi)
train['month_cycle_1']=train['month'].apply(lambda x: np.sin((x - 1) / 12 * np.pi) * np.pi)
train['hour_cycle_1']=train['hour'].apply(lambda x: np.sin((x) / 24 * np.pi) * np.pi)
train['period_cycle_1']=train['period'].apply(lambda x: np.sin((x - 1) / 7 * np.pi) * np.pi)

train['day_cycle_2']=train['day'].apply(lambda x: np.cos((x - 1) / 31 * np.pi) * np.pi + np.pi)
train['month_cycle_2']=train['month'].apply(lambda x: np.cos((x - 1) / 12 * np.pi) * np.pi + np.pi)
train['hour_cycle_2']=train['hour'].apply(lambda x: np.cos((x) / 24 * np.pi) * np.pi + np.pi)
train['period_cycle_2']=train['period'].apply(lambda x: np.cos((x - 1) / 7 * np.pi) * np.pi + np.pi)

test['day_cycle_1']=test['day'].apply(lambda x: np.sin((x - 1) / 31 * np.pi) * np.pi)
test['month_cycle_1']=test['month'].apply(lambda x: np.sin((x - 1) / 12 * np.pi) * np.pi)
test['hour_cycle_1']=test['hour'].apply(lambda x: np.sin((x) / 24 * np.pi) * np.pi)
test['period_cycle_1']=test['period'].apply(lambda x: np.sin((x - 1) / 7 * np.pi) * np.pi)

test['day_cycle_2']=test['day'].apply(lambda x: np.cos((x - 1) / 31 * np.pi) * np.pi + np.pi)
test['month_cycle_2']=test['month'].apply(lambda x: np.cos((x - 1) / 12 * np.pi) * np.pi + np.pi)
test['hour_cycle_2']=test['hour'].apply(lambda x: np.cos((x) / 24 * np.pi) * np.pi + np.pi)
test['period_cycle_2']=test['period'].apply(lambda x: np.cos((x - 1) / 7 * np.pi) * np.pi + np.pi)

#train=train.to_csv("/Users/chongshanxie/Desktop/2020Fall/MSBD5001/kaggle/data/final_train.csv",index=False)
#test=test.to_csv("/Users/chongshanxie/Desktop/2020Fall/MSBD5001/kaggle/data/final_test.csv",index=False)

#train=pd.read_csv("/Users/chongshanxie/Desktop/2020Fall/MSBD5001/kaggle/data/final_train.csv")
#test=pd.read_csv("/Users/chongshanxie/Desktop/2020Fall/MSBD5001/kaggle/data/final_test.csv")

#### The next part is model building and predicting part

X=train

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=1)

## The parameter choosing part
paramGrid = {
    'colsample_bytree': [0.6, 0.7, 0.8],
    'max_depth': [7, 8, 9, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [1500, 1750, 2000],
    'subsample': [0.6, 0.8],
    }


regXGB = XGBRegressor()

rsc = GridSearchCV(estimator=regXGB, param_grid=paramGrid, scoring='r2', cv=5, n_jobs=-1)
grid_result = rsc.fit(X_train, y_train)

best_params = grid_result.best_params_
print(grid_result.best_params_)
#best_params={'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 1750, 'subsample': 0.8,'seed':1130}


## The model building and training part

regXGB = XGBRegressor(**best_params)
regXGB.fit(X_train, y_train)
y_predicted = regXGB.predict(X_test)

print(mean_squared_error(y_test,y_predicted))

## Making the prediction

myspeed= regXGB.predict(test)
print(myspeed)

my_submission = pd.DataFrame({'id': data_test.id, 'speed': myspeed})
my_submission.to_csv('Submission_v3.1.csv', index=False)




