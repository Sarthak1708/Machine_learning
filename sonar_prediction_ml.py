import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv(r"D:\sonar_data.csv")
print(df)

print(df.head())

print(df.shape)

print(df.describe())

value_counts = df[df.columns[60]].value_counts()
print("Counts of Mines (M) and Rocks (R):\n", value_counts)

group_means = df.groupby(df.columns[60]).mean()
print(group_means)

X = df.drop(df.columns[60], axis=1)
Y = df[df.columns[60]]
print(X)
print(Y)

# training and testing the data 

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
print(X.shape,X_train.shape,X_test.shape)

print(X_train)
print(Y_train)

# initialize and train logistic model
model=LogisticRegression()

model.fit(X_train,Y_train)

# Evaluate accuracy on training and test data

X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print('Accuracy on training data:',training_data_accuracy)

X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('Accuracy on test data:',test_data_accuracy)

# Data input
input_data=(0.0423,0.0321,0.0709,0.0108,0.1070,0.0973,0.0961,0.1323,0.2462,0.2696,0.3412,0.4292,0.3682,0.3940,0.2965,0.3172,0.2825,0.3050,0.2408,0.5420,0.6802,0.6320,0.5824,0.6805,0.5984,0.8412,0.9911,0.9187,0.8005,0.6713,0.5632,0.7332,0.6038,0.2575,0.0349,0.1799,0.3039,0.4760,0.5756,0.4254,0.5046,0.7179,0.6163,0.5663,0.5749,0.3593,0.2526,0.2299,0.1271,0.0356,0.0367,0.0176,0.0035,0.0093,0.0121,0.0075,0.0056,0.0021,0.0043,0.0017)

# Convert input data to numpy array and reshape

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make prediction

prediction = model.predict(input_data_reshaped)

print(prediction)

if(prediction[0]=='R'):
    print('This is rock')
else:
    print('This is mine')