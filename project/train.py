#importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# naive bayes using multinomail distribution
class NaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha 

    def fit(self, X_train, y_train):
        m, n = X_train.shape
        self._classes = np.unique(y_train)
        n_classes = len(self._classes)

    # init: Prior & Likelihood
        self._priors = np.zeros(n_classes)
        self._likelihoods = np.zeros((n_classes, n))

    # Get Prior and Likelihood
        for idx, c in enumerate(self._classes):
            X_train_c = X_train[c == y_train]
            self._priors[idx] = X_train_c.shape[0] / m 
            self._likelihoods[idx, :] = ((X_train_c.sum(axis=0)) + self.alpha) / (np.sum(X_train_c.sum(axis=0) + self.alpha))


    def predict(self, X_test):
        return [self._predict(x_test) for x_test in X_test]

    def _predict(self, x_test):
    # Calculate posterior for each class
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior_c = np.log(self._priors[idx])
            likelihoods_c = self.calc_likelihood(self._likelihoods[idx,:], x_test)
            posteriors_c = np.sum(likelihoods_c) + prior_c
            posteriors.append(posteriors_c)

        return self._classes[np.argmax(posteriors)]

    def calc_likelihood(self, cls_likeli, x_test):
        return np.log(cls_likeli) * x_test

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.sum(y_pred == y_test)/len(y_test)


#import dataset
df = pd.read_csv("spam.csv",encoding="latin-1")
#remove unnecessary colums
df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1,inplace=True)
#giving column names
df= df.rename(columns={"v1":"label","v2":"text"})
#changing label with spam=1 and ham=0
df['label'] = df['label'].replace("spam",1)
df['label'] = df['label'].replace("ham",0)
# dataset
df.head()

#data preprocessing

import nltk  #importing nltk to download stop words
import re   
nltk.download("stopwords")  #downloading stopwords

from nltk.corpus import stopwords #import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()

#preprocessing data
def preprocessing():
    
    for i in range(0,len(df['text'])):
        temp = re.sub('[^a-zA-Z]',' ',df['text'][i])
        temp =temp.lower()
    
        temp =temp.split()
        temp = [ps.stem(word) for word in temp if word not in stopwords.words("english")]
        temp = " ".join(temp)
        df['text'] = df['text'].replace(df['text'][i],temp)
preprocessing()
print("dataset after preprocessing")
df.head()

#splitting dataset to 80% training and 20% testing

from sklearn.model_selection import train_test_split

X_train,x_test,Y_train, y_test = train_test_split(df['text'],df['label'],test_size=0.2,random_state=123)

#X_train have messages and Y_train have classes

#barchart and pie graphs checking wether data is divided into equal spam and non-spam

# count messages in spam and non-spam
x, y =Y_train.value_counts()
x2,y2 =y_test.value_counts()

tot_train = len(X_train)
tot_test = len(x_test)

def plot_bar(x,y,x2,y2):
    
    plt.bar(["Non-spam","spam"],[x,y],color=["blue","red"])
    plt.title("Training-dataset")
    plt.show()
    plt.bar(["Non-spam","spam"],[x2,y2],color=["blue","red"])
    plt.title("Testing-dataset")
    plt.show()

#plot_bar(x,y,x2,y2)


percentx = x*100/tot_train
percenty = y*100/tot_train

percentx2= x2*100/tot_test
percenty2=y2*100/tot_test

def pie_graph(percentx,percenty,percentx2,percenty2):
    
    plt.pie([percentx,percenty],labels=["non-spam","spam"],autopct='%1.2f%%')

    plt.legend(labels=["non-spam","spam"])

    plt.title("Training-dataset")

    plt.figure(0)
    
    plt.pie([percentx2,percenty2],labels=["non-spam","spam"],autopct='%1.2f%%')
    
    plt.title("Testing-dataset")
    
    plt.legend(labels=["non-spam","spam"])
    
    plt.figure(1)
    
    plt.show()
#pie_graph(percentx,percenty,percentx2,percenty2)
#feature extraction

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
#fitting training data such that cv gets unique words
cv.fit(X_train)

#forming vectors for training data

train_cv = cv.transform(X_train)

#forming vectors for testing data

test_cv = cv.transform(x_test)

print(len(str(cv.get_feature_names())+" are the total unique owrds present in training data"))

#total unique words


def freq_matrix(train_cv):
    
    frequency_matrix = pd.DataFrame(train_cv.toarray(),columns = cv.get_feature_names())
    return frequency_matrix
print(freq_matrix(train_cv))#printing frequency table of each message


#training the classifier using gaussian distribution

from sklearn.naive_bayes import MultinomialNB

#nb is using inbuilt classifier
nb = MultinomialNB()
nb.fit(train_cv.toarray(),Y_train)

#nb2 = classifier class NaiveBayes
nb2 = NaiveBayes()
nb2.fit(train_cv.toarray(),Y_train)

#testing using test vectors
predictions = nb.predict(test_cv.toarray())
predictions2 = nb2.predict(test_cv.toarray())

print("Score of nb for training set nb2: "+str(nb2.score(train_cv.toarray(),Y_train)))

print("Score of nb for testing set nb2: "+str(nb2.score(train_cv.toarray(),Y_train)))

#finding accuracy 

from sklearn.metrics import accuracy_score
print("accuracy score of nb2 {}%".format(accuracy_score(y_test,predictions2)*100))

from sklearn.metrics import precision_score,recall_score


print("Recall: {:.2f}%".format(100 * recall_score(y_test, predictions2)))

print("precision: {:.2f}%".format(100 * precision_score(y_test, predictions2)))
#error analysis using confusion matrix

from sklearn.metrics import confusion_matrix


print("confusion matrix")
print(confusion_matrix(y_test,predictions2))

#plotting confusion matrix

from sklearn.metrics import plot_confusion_matrix


def plot_matrix(nb,test_cv,y_test):
    matrix = plot_confusion_matrix(nb,test_cv.toarray(),y_test,cmap=plt.cm.Blues,)
    matrix.ax_.set_title("confusion-Matrix")
    plt.xlabel("predicted label")
    plt.ylabel("True label")
    plt.gcf().set_size_inches(10,6)
    plt.show()
#plot_matrix(nb,test_cv,y_test)

##saving the current model for further use

import pickle

import joblib

#for saving the model


joblib.dump(nb, 'NB_spam_model.pkl') 
joblib.dump(cv, 'cv.pkl')
joblib.dump(nb2,'NB_scratch_model.pkl')
print('model saved successfully')


#preprocessing the input
def preprocess(msg):
    txt = re.sub('[^A-Za-z]'," ",msg)
    txt = txt.lower()
    txt = txt.split()
    
    txt = [ps.stem(word) for word in txt if word not in stopwords.words("english")]
    mesg = ' '.join(txt)
    return mesg

##testing against custom input
def custom_input():
    
    msg=input('enter message')
    msg = preprocess(msg)
    msg_cv = cv.transform([msg])
    pred2 =nb2.predict(msg_cv.toarray())
    print(pred2)
custom_input()