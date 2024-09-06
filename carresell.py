from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn import metrics
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
df = pd.read_csv('E:/2021 projects/truevolts/car reselling/train-data.csv')
df1=df.fillna(0)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le
dfle = df1.copy()
dfle
dfle.Name = le.fit_transform(dfle.Name)
dfle.Location = le.fit_transform(dfle.Location)
dfle.Fuel_Type = le.fit_transform(dfle.Fuel_Type)
dfle.Transmission = le.fit_transform(dfle.Transmission)
dfle.Owner_Type = le.fit_transform(dfle.Owner_Type)
#dfle.Mileage = le.fit_transform(dfle.Mileage)
#dfle.Engine = le.fit_transform(dfle.Engine)
dfle
print(dfle)
X = dfle[['Name','Location','Year','Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Seats']].values
X
y = dfle.Price.values
y
atest=[[1200,9,2010,72000,0,1,0,5.0]]
#train_test separation
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
from sklearn.linear_model import LinearRegression
linear_clf = LinearRegression()
linear_clf.fit(X_train, y_train)
pred = linear_clf.predict(X_test)
pred1 = linear_clf.predict(atest)
print(pred)
print(pred1)