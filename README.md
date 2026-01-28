# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages.
2.Assigning hours to x and scores to y.
3.Plot the scatter plot.
4.Use mse,rmse,mae formula to find the values. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: N.DARSHINI 
RegisterNumber:  212225230200
*/
# IMPORT REQUIRED PACKAGE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```

## Output:
<img width="244" height="335" alt="image" src="https://github.com/user-attachments/assets/897ddac0-d3f8-4d36-9053-4282e58abc51" />
<img width="594" height="515" alt="image" src="https://github.com/user-attachments/assets/11360e0a-0ae0-4249-bc34-ee1442319d61" />
<img width="748" height="706" alt="image" src="https://github.com/user-attachments/assets/adc2017d-dc2e-48a3-aa06-8feedb9954b1" />
<img width="762" height="731" alt="image" src="https://github.com/user-attachments/assets/25ec7659-b92c-46b1-b3c1-766535aae46b" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
