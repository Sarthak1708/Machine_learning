import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm 

loan_data=pd.read_csv(r"D:\loan_dataset.csv")
print(loan_data.head())

# number of rows and column
print(loan_data.shape)

# getting some mathemetical information
print(loan_data.describe()) 

# finding null values
print(loan_data.isnull().sum())

# dropping the missing values
loan_data=loan_data.dropna()
print(loan_data)

# replacing loan_status
print(loan_data.replace({"Loan_Status":{"N":0,"Y":1}},inplace=True))

# dependent column value count
print(loan_data['Dependents'].value_counts())

# replacing 3+ to 4
loan_data=loan_data.replace(to_replace='3+',value=4)
print(loan_data['Dependents'].value_counts())

# data visualisation

# plotting count plot for education and loan status
sns.countplot(x='Education',hue='Loan_Status',data=loan_data)
plt.show()

# plotting count plot for maritial status and loan status
sns.countplot(x='Married',hue='Loan_Status',data=loan_data)
plt.show()

# convert categorical data to numerical values
print(loan_data.replace({"Gender":{"Male":1,"Female":0},"Married":{"Yes":1,"No":0},"Self_Employed":{"No":0,"Yes":1},
                   "Property_Area":{"Rural":0,"Urban":1,"Semiurban":2},"Education":{"Graduate":1,"Not Graduate":0}},inplace=True))

print(loan_data.shape)
# seperating the data and label
X=loan_data.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y=loan_data['Loan_Status']
print(X)
print(Y)

# train test split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)

# training the model:SVM
classifier=svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)

# # accuracy score of train and test data
X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print('Accuracy score of trianing data:',training_data_accuracy)

X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('Accuracy score of test data:',test_data_accuracy)

# making prediction model
input_data=(1,1,4,1,0,3036,2504,158,360,0,1)

# changing input array to numpy array
input_data_as_numpy_array=np.asarray(input_data)

prediction=classifier.predict(input_data_as_numpy_array.reshape(1,-1))
print(prediction)

if(prediction[0]==0):
    print('Loan not approved')
else:
    print('Loan approved')
