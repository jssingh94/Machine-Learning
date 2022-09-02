import pandas as pd
import numpy as np
import joblib 
df=pd.read_csv(r'C:\Users\Dell\Downloads\insurance.csv')

#1 check na values
print(df.isna().sum())

0#2 separate the dependent and independent variables 
x=df.iloc[:,: -1]
y=df.iloc[:,-1]

#3 cleaning 
#print(df.info())

#4 Encoding
# Label Encoding
from sklearn.preprocessing import LabelEncoder as Le
gender=Le()
x["gender"]=gender.fit_transform(x["gender"])
joblib.dump(gender,'gender.joblib')

smoker=Le()
x["smoker"]=smoker.fit_transform(x["smoker"])
joblib.dump(smoker,'smoker.joblib')

#One HOt Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer([('Encoding',OneHotEncoder(),[5])],remainder='passthrough')
x=ct.fit_transform(x)
joblib.dump(ct,'ct.joblib')

#5 Scaling
from sklearn.preprocessing import StandardScaler as SC
sc=SC()
x=sc.fit_transform(x)
joblib.dump(sc,'sc.joblib')

#6 Train test 
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,test_size=0.2,random_state=0)

#7 Training and testing of data set
#Linear Regression
# from sklearn.linear_model import LinearRegression as LR
# regressor=LR()                                           #.........@  >>>>>>fit_intercept=False pass this parameterr into LR()
# regressor.fit(x_train,y_train)

# # #testing
# y_pred=regressor.predict(x_test)

# # #8 Evaluation of test
# # #Metrics
# from sklearn.metrics import r2_score
# print(r2_score(y_test,y_pred))
# #r2 score tells us how much variance of dependent variable is explained by independent columns
# # its value lies between -infinty to +1. If accuracy is lowe than tuning method is used.
# #Tuning can be done in three ways 
# #1 Tuning of parameters in @ code
# #2 Change strategy 
# # Support Vector Regression

# from sklearn.svm import SVR
# regressor=SVR(C=10000)                                           #........$
# regressor.fit(x_train,y_train)
# # Testing
# y_pred=regressor.predict(x_test)
# #Evaluation
# from sklearn.metrics import r2_score
# print(r2_score(y_test,y_pred))

#Again tuning of parameters in code $



# 16.05-2022

# from sklearn.tree import DecisionTreeRegressor as LR

# regressor=LR()

# regressor.fit(x_train,y_train)

# y_pred=regressor.predict(x_test)
# #Evaluation
# from sklearn.metrics import r2_score
# print(r2_score(y_test,y_pred))

#  insurance Data


#using randomforest strategy
from sklearn.ensemble import RandomForestRegressor as LR
regressor=LR()

regressor.fit(x_train,y_train)
joblib.dump(regressor,'regressor.joblib')

y_pred=regressor.predict(x_test)

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))






