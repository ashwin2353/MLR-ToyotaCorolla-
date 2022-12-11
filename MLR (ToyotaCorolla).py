# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 12:22:00 2022

@author: ashwi
"""
import pandas as pd
df = pd.read_csv("ToyotaCorolla (1).csv",encoding='latin1')
df.head()
df.dtypes
df.shape

# Blanks
df.isnull().sum()
# finding duplicate rows
df.duplicated()
df[df.duplicated()] # hence no duplicates between the rows

# finding du`plicate columns
df.columns.duplicated() # hence no duplicates between the column

#==================================================
# dropping the variables
df.drop("Id",axis=1,inplace=True)
df.drop("Model",axis=1,inplace=True)
df.drop("Mfg_Month",axis=1,inplace=True)
df.drop("Mfg_Year",axis=1,inplace=True)
df.drop("Fuel_Type",axis=1,inplace=True)
df.drop("Met_Color",axis=1,inplace=True)
df.drop("Color",axis=1,inplace=True)
df.drop("Automatic",axis=1,inplace=True)
df.drop("Cylinders",axis=1,inplace=True)
df.dtypes

df.drop(df.iloc[:,9:],axis=1,inplace=True)
df.dtypes
df.head()
df.shape

#=================================================
# Data visualization
# Histogram
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

for col in df:
    print(col)
    print(skew(df[col]))
    
    plt.figure()
    sns.distplot(df[col])
    plt.show()

#==================================================
# scatter plot
plt.scatter(df["Price"],df["Age_08_04"],color = "black")
# in this variables they have -ve relationship
plt.scatter(df["Price"],df["KM"],color = "black")
# in this variables they have -ve relationship
plt.scatter(df["Price"],df["HP"],color = "black")
# they have no relationship
plt.scatter(df["Price"],df["cc"],color = "black")
# they have no relationship
plt.scatter(df["Price"],df["Doors"],color = "black")
# they have no relationship
plt.scatter(df["Price"],df["Gears"],color = "black")
# they have no relationship
plt.scatter(df["Price"],df["Quarterly_Tax"],color = "black")
# they have no relationship
plt.scatter(df["Price"],df["Weight"],color = "black")
# they have no relationship
#============================================================
# boxplot
df.boxplot(column="Price",vert=False)
import numpy as np
Q1 = np.percentile(df["Price"],25)
Q2 = np.percentile(df["Price"],50)
Q3 = np.percentile(df["Price"],75)
IQR = Q3 - Q1
LW = Q1 - (2.5*IQR)
UW = Q3 + (2.5*IQR)
df[(df["Price"]<LW) | (df["Price"]>UW)]
len(df[(df["Price"]<LW) | (df["Price"]>UW)])
# outlaires are 34
df["Price"]=np.where(df["Price"]>UW,UW,np.where(df["Price"]<LW,LW,df["Price"]))
len(df[(df["Price"]<LW) | (df["Price"]>UW)])
# outlaires are 0

df.boxplot(column="Age_08_04",vert=False)
import numpy as np
Q1 = np.percentile(df["Age_08_04"],25)
Q2 = np.percentile(df["Age_08_04"],50)
Q3 = np.percentile(df["Age_08_04"],75)
IQR = Q3 - Q1
LW = Q1 - (2.5*IQR)
UW = Q3 + (2.5*IQR)
df[(df["Age_08_04"]<LW) | (df["Age_08_04"]>UW)]
len(df[(df["Age_08_04"]<LW) | (df["Age_08_04"]>UW)])
# outlaires are 0


df.boxplot(column="KM",vert=False)
import numpy as np
Q1 = np.percentile(df["KM"],25)
Q2 = np.percentile(df["KM"],50)
Q3 = np.percentile(df["KM"],75)
IQR = Q3 - Q1
LW = Q1 - (2.5*IQR)
UW = Q3 + (2.5*IQR)
df[(df["KM"]<LW) | (df["KM"]>UW)]
len(df[(df["KM"]<LW) | (df["KM"]>UW)])
# outlaires are 12
df["KM"]=np.where(df["KM"]>UW,UW,np.where(df["KM"]<LW,LW,df["KM"]))
len(df[(df["KM"]<LW) | (df["KM"]>UW)])
# outlaires are 0


df.boxplot(column="HP",vert=False)
import numpy as np
Q1 = np.percentile(df["HP"],25)
Q2 = np.percentile(df["HP"],50)
Q3 = np.percentile(df["HP"],75)
IQR = Q3 - Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["HP"]<LW) | (df["HP"]>UW)]
len(df[(df["HP"]<LW) | (df["HP"]>UW)])
# outlaires are 11
df["HP"]=np.where(df["HP"]>UW,UW,np.where(df["HP"]<LW,LW,df["HP"]))
len(df[(df["HP"]<LW) | (df["HP"]>UW)])
# outlaires are 0


df.boxplot(column="Doors",vert=False)
import numpy as np
Q1 = np.percentile(df["Doors"],25)
Q2 = np.percentile(df["Doors"],50)
Q3 = np.percentile(df["Doors"],75)
IQR = Q3 - Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["Doors"]<LW) | (df["Doors"]>UW)]
len(df[(df["Doors"]<LW) | (df["Doors"]>UW)])
# outlaires are 0


df.boxplot(column="Gears",vert=False)
import numpy as np
Q1 = np.percentile(df["Gears"],25)
Q2 = np.percentile(df["Gears"],50)
Q3 = np.percentile(df["Gears"],75)
IQR = Q3 - Q1
LW = Q1 - (2.5*IQR)
UW = Q3 + (2.5*IQR)
df[(df["Gears"]<LW) | (df["Gears"]>UW)]
len(df[(df["Gears"]<LW) | (df["Gears"]>UW)])
# outlaires are 46
df["Gears"]=np.where(df["Gears"]>UW,UW,np.where(df["Gears"]<LW,LW,df["Gears"]))
len(df[(df["Gears"]<LW) | (df["Gears"]>UW)])
# outlaires are 0

df.boxplot(column="Quarterly_Tax",vert=False)
import numpy as np
Q1 = np.percentile(df["Quarterly_Tax"],25)
Q2 = np.percentile(df["Quarterly_Tax"],50)
Q3 = np.percentile(df["Quarterly_Tax"],75)
IQR = Q3 - Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["Quarterly_Tax"]<LW) | (df["Quarterly_Tax"]>UW)]
len(df[(df["Quarterly_Tax"]<LW) | (df["Quarterly_Tax"]>UW)])
# outlaires are 224
df["Quarterly_Tax"]=np.where(df["Quarterly_Tax"]>UW,UW,np.where(df["Quarterly_Tax"]<LW,LW,df["Quarterly_Tax"]))
len(df[(df["Quarterly_Tax"]<LW) | (df["Quarterly_Tax"]>UW)])
# outlaires are 0

df.boxplot(column="Weight",vert=False)
import numpy as np
Q1 = np.percentile(df["Weight"],25)
Q2 = np.percentile(df["Weight"],50)
Q3 = np.percentile(df["Weight"],75)
IQR = Q3 - Q1
LW = Q1 - (2.5*IQR)
UW = Q3 + (2.5*IQR)
df[(df["Weight"]<LW) | (df["Weight"]>UW)]
len(df[(df["Weight"]<LW) | (df["Weight"]>UW)])
# outlaires are 34
df["Weight"]=np.where(df["Weight"]>UW,UW,np.where(df["Weight"]<LW,LW,df["Weight"]))
len(df[(df["Weight"]<LW) | (df["Weight"]>UW)])
# outlaires are 0
#======================================================================================
# Standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
df[["Price"]] = SS.fit_transform(df[["Price"]])
df[["Age_08_04"]] = SS.fit_transform(df[["Age_08_04"]])
df[["KM"]] = SS.fit_transform(df[["KM"]])
df[["HP"]] = SS.fit_transform(df[["HP"]])
df[["cc"]] = SS.fit_transform(df[["cc"]])
df[["Doors"]] = SS.fit_transform(df[["Doors"]])
df[["Gears"]] = SS.fit_transform(df[["Gears"]])
df[["Quarterly_Tax"]] = SS.fit_transform(df[["Quarterly_Tax"]])
df[["Weight"]] = SS.fit_transform(df[["Weight"]])

df.dtypes
df.head()
df.corr
df.corr().to_csv("X.list.csv")
#=========================================================================================================
# Splitting the variables as X and Y

Y = df.iloc[:,:1]
# X = df.iloc[:,1:] # Model-1
# X = df[["Age_08_04","Weight","HP","cc","Doors","Gears"]] # Model-2
X = df[["KM","Quarterly_Tax"]] # Model-3
#=================================================================================================
# Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=32)
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape

#===========================================================================================
# Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train)

# B0
LR.intercept_

# B1
LR.coef_

# predictions
Y_pred_train = LR.predict(X_train)
Y_pred_train
 
Y_pred_test = LR.predict(X_test)
Y_pred_test

# Marics
from sklearn.metrics import mean_squared_error,r2_score
Training_Error = mean_squared_error(Y_train, Y_pred_train)
Testing_Error = mean_squared_error(Y_test, Y_pred_test)

print("Training Error :",Training_Error.round(3))
print("Testing Error :",Testing_Error.round(3))

import numpy as np
print("Root Mean Squared Error :", np.sqrt(Training_Error).round(3))
print("Root Mean Squared Error :", np.sqrt(Testing_Error).round(3))

r2 = r2_score(Y_train,Y_pred_train)
print("R square :", r2.round(3))

r2 = r2_score(Y_test,Y_pred_test)
print("R square :", r2.round(3))
#===========================================================
# validation set approch 
TrE = []
TsE = []
for i in range(1,101):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    LR.fit(X_train,Y_train)
    Y_pred_train = LR.predict(X_train)
    Y_pred_test = LR.predict(X_test)
    TrE.append(mean_squared_error(Y_train,Y_pred_train))
    TsE.append(mean_squared_error(Y_test,Y_pred_test))

print(TrE)
print(TsE)

import numpy as np
np.mean(TrE)
np.mean(TsE)
#=====================================================================
# K-fold cross-validation
from sklearn.model_selection import KFold,cross_val_score
kfold = KFold(n_splits=10)
LR = LinearRegression()
train_result = abs(cross_val_score(LR, X_train, Y_train, cv=kfold,scoring="neg_mean_squared_error"))
test_result = abs(cross_val_score(LR, X_test, Y_test, cv=kfold,scoring="neg_mean_squared_error"))
train_result
test_result

train_result.mean()
test_result.mean()

#=====================================================================
# LOOCV
from sklearn.model_selection import LeaveOneOut,cross_val_score
loocv = LeaveOneOut()
LR =LinearRegression()
train_result1 = abs(cross_val_score(LR,X_train,Y_train,cv=loocv,scoring="neg_mean_squared_error"))
test_result1 = abs(cross_val_score(LR,X_test,Y_test,cv=loocv,scoring="neg_mean_squared_error"))
train_result1
test_result1

train_result1.mean()
test_result1.mean()
# therefore by see all the model validation techniques i did't notice any drastic reduced in error's so i decided that not to go with any validation techniques

#===========================================================
import statsmodels.api as sma
Y_new =sma.add_constant(X)
lm2 = sma.OLS(Y,Y_new).fit()
lm2.summary()

# Therefore from above 3 models, model-2 is giving best results but P-value is getting more then 0.05 
# To check model is best or not VIF is doing
####################################  VIF  ####################################
import pandas as pd
df = pd.read_csv("ToyotaCorolla (1).csv",encoding='latin1')
df.head()

# dropping the variables
df.drop("Id",axis=1,inplace=True)
df.drop("Model",axis=1,inplace=True)
df.drop("Mfg_Month",axis=1,inplace=True)
df.drop("Mfg_Year",axis=1,inplace=True)
df.drop("Fuel_Type",axis=1,inplace=True)
df.drop("Met_Color",axis=1,inplace=True)
df.drop("Color",axis=1,inplace=True)
df.drop("Automatic",axis=1,inplace=True)
df.drop("Cylinders",axis=1,inplace=True)
df.dtypes

df.drop(df.iloc[:,9:],axis=1,inplace=True)
df.dtypes
df.head()

# loading the data
Y = df.iloc[:,:1]
X = df[["KM","Quarterly_Tax"]]

# import linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression().fit(X,Y)

# predict with our model coeffcients
Y_pred = lm.predict(X)

# monual calculations
import pandas as pd
import numpy as np
RSS = np.sum((Y_pred - Y)**2) # Residual sum of squares
Y_mean = np.mean(Y)
TSS = np.sum((Y - Y_mean)**2)
R2 = 1-(RSS/TSS)
print("R2 :", R2)

VIF = 1/(1-R2)
print("VIF value :",VIF)

# Therefore for model-2 VIF value got (6.144 ) and for model-3 VIF value we got (1.921) 
# to decide which model is excellent our VIF value should be less then 5

#########################################################################################
################################### final model #########################################

Y = df.iloc[:,:1]
X = df[["KM","Quarterly_Tax"]] # Model-3


# Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=32)
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape

# Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train)

# B0
LR.intercept_

# B1
LR.coef_

# predictions
Y_pred_train = LR.predict(X_train)
Y_pred_train
 
Y_pred_test = LR.predict(X_test)
Y_pred_test

# Marics
from sklearn.metrics import mean_squared_error,r2_score
Training_Error = mean_squared_error(Y_train, Y_pred_train)
Testing_Error = mean_squared_error(Y_test, Y_pred_test)

print("Training Error :",Training_Error.round(3))
print("Testing Error :",Testing_Error.round(3))

import numpy as np
print("Root Mean Squared Error :", np.sqrt(Training_Error).round(3))
print("Root Mean Squared Error :", np.sqrt(Testing_Error).round(3))

r2 = r2_score(Y_train,Y_pred_train)
print("R square :", r2.round(3))

r2 = r2_score(Y_test,Y_pred_test)
print("R square :", r2.round(3))



