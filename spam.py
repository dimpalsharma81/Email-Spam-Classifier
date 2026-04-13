import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('mail_data.csv')
print(df)
data = df.where((pd.notnull(df)), '')
data.head(10)
data.info()
data.shape
print(data.columns)
data.loc[data['Category'] == 'spam','Category'] = '0'
data.loc[data['Category'] == 'ham','Category'] = '1'
X = data['Message']
Y = data['Category']
Y = Y.astype('int')
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state = 3)
print(X.shape)
print(X_train.shape)
print(X_test.shape)
print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)
feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english',lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
print(X_train)
print(X_train_features)
model = LogisticRegression()
model.fit(X_train_features, Y_train)
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Acc on training data : ', accuracy_on_training_data)
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Acc on test data : ', accuracy_on_test_data)
st.title("Email Spam Classifier")

input_sms = st.text_area("Enter the messager")
if st.button('predict'):
    vector_input = feature_extraction.transform([input_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Not Spam (Ham)")
    else:
        st.header("Spam Message")


    

    