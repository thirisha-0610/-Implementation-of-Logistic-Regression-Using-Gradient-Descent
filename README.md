# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and load the dataset.

2.Define X and Y array.

3.Define a function for costFunction,cost and gradient.

4.Define a function to plot the decision boundary.

5.Define a function to predict the Regression value

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Thirisha A
RegisterNumber:  212223040228
*/
```
```
import pandas as pd
import numpy as np
dataset=pd.read_csv("Placement_Data.csv")
dataset
```
![435556536-fcfab480-b3ca-45c5-9bf1-3d65ba577631](https://github.com/user-attachments/assets/2e6e5881-139d-40fb-ba4a-ce1b0ecc12ac)

```
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```
![435556653-d9965e24-5e0b-4053-aa0e-5301f8a56a10](https://github.com/user-attachments/assets/9bd12016-db80-49c3-9fad-67839042769e)

```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```
![435556760-cc263ecf-94ff-4bd4-aa65-512a633a185b](https://github.com/user-attachments/assets/1eae5491-f08b-4290-b08a-a33d79a35116)

```
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
```
![435556927-db3b870e-8f62-4b46-9f83-ecb0bc0f357e](https://github.com/user-attachments/assets/47c65061-da95-4595-9368-13c643ceda4d)

```
theta = np.random.randn(X.shape[1])
y =Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h= sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations = 1000)
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred

y_pred = predict(theta,X)
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)
```
![435557037-cabb0090-a308-4bed-8622-ef8fef9d13fd](https://github.com/user-attachments/assets/9db05a46-9d7b-4e14-aff4-3e8d25237e02)

```
print(y_pred)
```
![435557187-3528b460-d655-434c-b14e-54e46dd9b6f5](https://github.com/user-attachments/assets/18d88877-541f-4cea-bfaf-af424746998d)
```
print(y)
```
![435557277-158953e8-1bd2-40fd-a4cc-57d42bbcf918](https://github.com/user-attachments/assets/59c4227d-72f3-42fa-9bfd-2efa566717b5)
```
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```
![435557414-4508311c-ea58-4c4b-b64e-71abed8e11cb](https://github.com/user-attachments/assets/a3e0e22c-8e4d-4aad-ba22-1eb0ac4f6606)
```
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```
![435557595-03b311b4-2506-4893-a341-433964dca8ed](https://github.com/user-attachments/assets/7333e2f3-93ce-4761-83b0-268411427528)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

