# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 08:48:38 2020

@author: Radha Kashyap
"""

import pandas as pd
df=pd.read_csv("news.csv")
df.head()
df.shape
df=df.set_index("Unnamed: 0")
df.head()

y=df.label
df.drop("label",axis=1)
y.head()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df["text"],
                              y,test_size=0.33,random_state=53)
X_train.head()


from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer=CountVectorizer(stop_words='english')
count_train=count_vectorizer.fit_transform(X_train)
count_test=count_vectorizer.transform(X_test)

print(count_train.A.shape)
print(X_train.shape)
print(len(count_vectorizer.get_feature_names()))

print(count_vectorizer.get_feature_names()[:10])
count_df=pd.DataFrame(count_train.A,
                      columns=count_vectorizer.get_feature_names())
count_df.isnull().sum(axis=1)
count_df.head()

#using multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()
from sklearn import metrics
clf.fit(count_train,y_train)
pred=clf.predict(count_test)
score=metrics.accuracy_score(y_test,pred)
print("accuracy : %0.3f" % score)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words="english",max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(X_train)
tfidf_test=tfidf_vectorizer.transform(X_test)
tfidf_df=pd.DataFrame(tfidf_vectorizer.get_feature_names())
tfidf_df.head()

clf=MultinomialNB()
clf.fit(tfidf_train,y_train)
pred=clf.predict(tfidf_test)
score=metrics.accuracy_score(y_test,pred)

#using Psiiveaggressiveclassifier
from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf=PassiveAggressiveClassifier(n_iter=50)
linear_clf.fit(tfidf_train,y_train)
pred=linear_clf.predict(tfidf_test)
score=metrics.accuracy_score(y_test,pred)
print("accuracy : %0.3f" % score)

linear_clf.fit(count_train,y_train)
pred=linear_clf.predict(count_test)
score=metrics.accuracy_score(y_test,pred)
print("accuracy : %0.3f" % score)


























