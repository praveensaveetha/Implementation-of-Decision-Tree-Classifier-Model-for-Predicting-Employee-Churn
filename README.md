# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Praveen V
RegisterNumber: 212222040121

import pandas as pd
data=pd.read_csv("dataset/Employee.csv")
data.head()

data.info()

data.isnull().sum()
data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evalution","number_project","average_montly_hours","time_spend_company","work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### Data Head
![head](https://github.com/A-Thiyagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707693/8d1459aa-5216-4dff-9821-dc42ba4993c8)

### Information:
![info](https://github.com/A-Thiyagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707693/34e37767-ae95-4f0c-b608-89f73e832036)

### Null dataset:
![null](https://github.com/A-Thiyagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707693/14090052-051a-44b6-921b-38b2b398928a)

### Value_counys:
![v_count](https://github.com/A-Thiyagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707693/44b8a5ce-bf50-40e2-8f8f-d704dc157bdc)

### Data Type Convertion:
![change to number](https://github.com/A-Thiyagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707693/0f3bfc5f-a98d-4fac-9eb4-4a26fc7e57c4)

### Data Info:
![data info](https://github.com/A-Thiyagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707693/1a3c3f83-57d7-4537-a438-468b58083138)

### Accuracy:
![accu](https://github.com/A-Thiyagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707693/0fbfb6a8-9209-4b87-91b8-078c68e302fb)

### Data Prediction:
![pred](https://github.com/A-Thiyagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707693/d9280463-01e2-41dc-8a35-1dfb67feb7c0)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
