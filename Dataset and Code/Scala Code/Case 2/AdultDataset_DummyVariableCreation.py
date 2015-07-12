%pylab inline
# Pandas package... using data frames in python
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from numpy import genfromtxt, savetxt
import random
df = pd.read_csv('C:\\Users\\anirudhbedre\\Downloads\\Adult dataset\\adult.data',header=None,skipinitialspace=True)
dummies = pd.get_dummies(df.iloc[:,1])
df = df.join(dummies)
df=df.drop(["?","Federal-gov"],axis=1)
dummies2 = pd.get_dummies(df.iloc[:,5])
df = df.join(dummies2)
df=df.drop("Never-married",axis=1)
dummies3 = pd.get_dummies(df.iloc[:,6])
df = df.join(dummies3)
df=df.drop("Adm-clerical",axis=1)
df=df.drop("?",axis=1)
dummies4 = pd.get_dummies(df.iloc[:,7])
df = df.join(dummies4)
df=df.drop("Not-in-family",axis=1)
dummies5 = pd.get_dummies(df.iloc[:,8])
df = df.join(dummies5)
df=df.drop("White",axis=1)
dummies6 = pd.get_dummies(df.iloc[:,9])
df = df.join(dummies6)
df=df.drop("Male",axis=1)
dummies7 = pd.get_dummies(df.iloc[:,13])
df = df.join(dummies7)
df=df.drop("Cuba",axis=1)
df=df.drop("?",axis=1)
df=df.drop([1,3,5,6,7,8,9,13],axis=1)
cols = df.columns.tolist()
#rearranging columns  -- shifted 'hour' column in front here
cols = [0,2,4,10,11,12,'Local-gov','Never-worked','Private','Self-emp-inc','Self-emp-not-inc','State-gov','Without-pay','Divorced','Married-AF-spouse','Married-civ-spouse','Married-spouse-absent','Separated','Widowed','Armed-Forces','Craft-repair','Exec-managerial','Farming-fishing','Handlers-cleaners','Machine-op-inspct','Other-service','Priv-house-serv','Prof-specialty','Protective-serv','Sales','Tech-support','Transport-moving','Husband','Other-relative','Own-child','Unmarried','Wife','Amer-Indian-Eskimo','Asian-Pac-Islander','Black','Other','Female','Cambodia','Canada','China','Columbia','Dominican-Republic','Ecuador','El-Salvador','England','France','Germany','Greece','Guatemala','Haiti','Holand-Netherlands','Honduras','Hong','Hungary','India','Iran','Ireland','Italy','Jamaica','Japan','Laos','Mexico','Nicaragua','Outlying-US(Guam-USVI-etc)','Peru','Philippines','Poland','Portugal','Puerto-Rico','Scotland','South','Taiwan','Thailand','Trinadad&Tobago','United-States','Vietnam','Yugoslavia',14]
df = df[cols]
df.to_csv(path_or_buf='adult_withdummies.csv')