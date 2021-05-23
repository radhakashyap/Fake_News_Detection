# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 08:48:38 2020

@author: Radha Kashyap
"""
#Importing necessary python Libraries
import panda as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import PassiveAggressiveClassifier

#Read the Dataset
df=pd.read_csv("news.csv")

#Get shape of 'df'
df.shape
#print First line of 'df'
df.head()

df=df.set_index("Unnamed: 0")
df.head()
df.drop("label",axis=1)
df.head()

#Reading the labels of data
y=df.label
y.head()

#splitting the dataset into training set and test set
X_train,X_test,y_train,y_test=train_test_split(df['text'],
                                               y,test_size=0.33,random_state=53)
X_train.head()
y_train.head()

#Initialize the CountVectorizer
count_vectorizer=CountVectorizer(stop_words='english')
#Fit and transform the train set and transform the test set
count_train=count_vectorizer.fit_transform(X_train)
count_test=count_vectorizer.transform(X_test)

#Intialize the TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7)
#Fit and transform the train set and transform the test set
tfidf_train=tfidf_vectorizer.fit_transform(X_train)
tfidf_test=tfidf_vectorizer.transform(X_test)
print(tfidf_test)

#Get the feature names
print(tfidf_vectorizer.get_feature_names()[-10 :])
print(count_vectorizer.get_feature_names()[0 :10])


count_df=pd.DataFrame(count_train.A,
                      columns=count_vectorizer.get_feature_names())
tfidf_df=pd.DataFrame(tfidf_train.A,
                     columns=tfidf_vectorizer.get_feature_names())

#Calculate the difference 
difference=set(count_df.columns)-set(tfidf_df.columns)
difference
set()
print(count_df.equals(tfidf_df))
count_df.head()

import matplotlib.pyplot as plt
def plot_confusion_matrix(cm,classes,normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print("normalized confusion matrix")
    else:
        print("confusion matrix without normalization")
    
    thresh=cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment="center", 
                 color="white" if cm[i,j]>thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('predicted label')
    
#using Multinomial Naive Bayes algorithm  on TfidfVecorizer   
clf=MultinomialNB()
clf.fit(tfidf_train,y_train)
pred=clf.predict(tfidf_test)

#calculate the score
score=accuracy_score(y_test,pred)
print("accuracy:%0.3f" % score)
cm=confusion_matrix(y_test,pred,labels=['FAKE' ,'REAL'])
#plotting the confusion matrix
plot_confusion_matrix(cm,classes=['FAKE','REAL'])    

#usinf Multinomial Naive Bayes on CountVectorizer
clf=MultinomialNB()
clf.fit(count_train,y_train)
pred=clf.predict(count_test)

#calculate the score
score=accuracy_score(y_test,pred)
print("accuracy:%3f" % score)
cm=confusion_matrix(y_test,pred,labels=['FAKE' ,'REAL'])
#plotting the confusion matrix 
plot_confusion_matrix(cm,classes=['FAKE','REAL']) 
  
#Using PassiveAggressiveClassifier Algorithm on TfidfVectorizer
linear_clf=PassiveAggressiveClassifier(n_iter=50)
linear_clf.fit(tfidf_train,y_train)
pred=linear_clf.predict(tfidf_test)
#calculate the score
score=accuracy_score(y_test,pred)
print("accuracy:%3f" % score)
cm=confusion_matrix(y_test,pred,labels=['FAKE' ,'REAL'])
#plotting the confusion matrix
plot_confusion_matrix(cm,classes=['FAKE','REAL'])    
  
#using PassiveAggressiveClassifier algorithm on CountVectorizer  
    
linear_clf.fit(count_train,y_train)
pred=linear_clf.predict(count_test)
#calculate the score
score=accuracy_score(y_test,pred)
print("accuracy : %0.3f" % score)
cm=confusion_matrix(y_test,pred,labels=['FAKE' ,'REAL'])
#plotting the confusion matrix
plot_confusion_matrix(cm,classes=['FAKE','REAL'])    
        
   
    
    









































