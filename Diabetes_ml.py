import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

diabetes_data=pd.read_csv("D:\diabetes.csv")
print(diabetes_data)
print(diabetes_data.head())
print(diabetes_data.shape)
print(diabetes_data.describe())
print(diabetes_data['Outcome'].value_counts())
print(diabetes_data.groupby('Outcome').mean())

X=diabetes_data.drop(columns='Outcome',axis=1)
Y=diabetes_data['Outcome']

print(X)
print(Y)

# Data Standardization
scaler=StandardScaler()
scaler.fit(X)

standardized_data=scaler.transform(X)
print(standardized_data)

X=standardized_data
Y=diabetes_data['Outcome']

print(X)
print(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)

# Training the model
classifier=svm.SVC(kernel='linear')

# Training the SVM
classifier.fit(X_train,Y_train)

# Modle valuation and accuracy score
X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print('Accuracy score of the training data:',training_data_accuracy)

X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('Accuracy score of the test data:',test_data_accuracy)

# Making a prediction model
input=(1,103,30,38,83,43.3,0.183,33)

input_data_as_numpy_array=np.asarray(input)

input_data_reshape=input_data_as_numpy_array.reshape(1,-1)

std_data=scaler.transform(input_data_reshape)
print(std_data)

prediction=classifier.predict(std_data)
print(prediction)

if(prediction[0]==0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')