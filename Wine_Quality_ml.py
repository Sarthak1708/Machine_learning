import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

wine_data=pd.read_csv(r"D:\winequality-red.csv")
print(wine_data.head())

print(wine_data.shape)

wine_data.isnull().sum()
print(wine_data)

print(wine_data.describe())

# data visualization

# number of values for qualtiy

sns.catplot(x='quality',data=wine_data,kind='count')
plt.show()

# volatile acidity vs quality 

plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='volatile acidity',data=wine_data)
plt.show()

# citric acid vs quality

plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='citric acid',data=wine_data)
plt.show()

# correlation

correlation=wine_data.corr()

# plotting a heatmap

plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')
plt.show()

# data processing
# separate the data and label

X=wine_data.drop('quality',axis=1)
Y=wine_data['quality'].apply(lambda y_value:1 if y_value>=7 else 0)
print(X)
print(Y)

# train test split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)

# model training :Random forest classifier

model=RandomForestClassifier()
model.fit(X_train,Y_train)

# accuracy on test data

X_test_prediction=model.predict(X_test)
test_accuracy_prediction=accuracy_score(X_test_prediction,Y_test)
print('Accuracy score on test data:',test_accuracy_prediction)

# making a predicitive model 

input_data=(7.8,0.58,0.02,2.0,0.073,9.0,18.0,0.9968,3.36,0.57,9.5)

# changing the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

# reshaping the data
input_data_reshape=input_data_as_numpy_array.reshape(1,-1)

prediciton=model.predict(input_data_reshape)
print(prediciton)

if(prediciton[0]==1):
    print('Good quality wine')
else:
    print('Bad quality wine')