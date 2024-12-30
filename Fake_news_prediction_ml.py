import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

print(stopwords.words('english'))

news_data = pd.read_csv(r"D:\train.csv")

print(news_data.head())

# 0 -> real news
# 1 -> fake news

print(news_data.shape)

print(news_data.isnull().sum())

# replacing null values with empty string 
news_data=news_data.fillna('')

# merging author name and title
news_data['content']=news_data['author']+'->'+news_data['title']
print(news_data['content'])

# seprating the data and label
X=news_data.drop(columns='label',axis=1)
Y=news_data['label']
print(X)
print(Y)

# stemming:process of reducing a word to its root word
port_stem=PorterStemmer()

def stemming(content):
    stemmed_content=re.sub('[^a-zA-Z]',' ',content)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=' '.join(stemmed_content)
    return stemmed_content

news_data['content']=news_data['content'].apply(stemming)
print(news_data['content'])

# separating the data and label
X=news_data['content'].values
Y=news_data['label'].values
print(X)
print(Y)
print(Y.shape)

# converting the textual data to numerical data
vectorizer=TfidfVectorizer()
vectorizer.fit(X)

X=vectorizer.transform(X)
print(X)

# spliting the data into training and testing 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

# training logistic regressing model
model=LogisticRegression()
model.fit(X_train,Y_train)

# accuracy score
X_train_predicition=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_predicition,Y_train)
print('accuracy score of training data:',training_data_accuracy)

X_test_predicition=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_predicition,Y_test)
print('accuracy score of test data:',test_data_accuracy)

# making a prediction model
X_new=X_test[0]
prediction=model.predict(X_new)
print(prediction)

if(prediction[0]==0):
    print('The news is real')
else:
    print('The news is fake')

print(Y_test[0])