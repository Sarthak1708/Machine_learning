import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# loading and performing basic analysis
gold_data=pd.read_csv(r"D:\gld_price_data.csv")
print(gold_data.head())
print(gold_data.shape)
print(gold_data.info())
print(gold_data.describe)
print(gold_data.isnull().sum())

gold_data.dropna(inplace=True)
gold_data['Date'] = pd.to_datetime(gold_data['Date'], errors='coerce') if 'Date' in gold_data.columns else gold_data['Date']

correlation = gold_data.select_dtypes(include=['float64', 'int64']).corr()

correlation=gold_data.corr()

# heatmap to understand correlation 
plt.figure(figsize=(8,8))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')
plt.show()

# correalation values of gold
print(correlation['GLD'])

# distribution of gold price
sns.displot(gold_data['GLD'],color='green')
plt.show()

X=gold_data.drop(['Date','GLD'],axis=1)
Y=gold_data['GLD']

print(X)
print(Y)

# splititng the data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

regressor=RandomForestRegressor(n_estimators=100)

# training the model
regressor.fit(X_train,Y_train)

# prediciton on test data
test_data_prediction=regressor.predict(X_test)
print(test_data_prediction)

# r squared error
error_score=metrics.r2_score(Y_test,test_data_prediction)
print("R squared error:",error_score)

# comapre actual and predicted values in plot
Y_test=list(Y_test)

plt.plot(Y_test,color='Blue',label='Actual Value')
plt.plot(test_data_prediction,color='green',label='Predicted value')
plt.title('Actual vs Predicted value')
plt.xlabel('Number of Value')
plt.ylabel('GLD price')
plt.legend()
plt.show()