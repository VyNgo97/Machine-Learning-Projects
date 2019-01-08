#!/usr/bin/env python
# coding: utf-8

# In[42]:


#Vy Ngo
#Project D: Text Analysis
import nltk
import pandas
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#download nltk once
#nltk.download() 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[48]:


#import csv file with special encoding
data = pandas.read_csv("spam.csv", delimiter=",", encoding="ISO-8859–1")
#print out the first message
print(data["content"][0])
rev_soup = BeautifulSoup(data["content"][0])
print("\n")
print(rev_soup.get_text())


# In[50]:


letters_only = re.sub("[^a-zA-Z]"," ",rev_soup.get_text())
print(letters_only)


# In[52]:


lower_case = letters_only.lower()
words = lower_case.split()
#print out the nltk stop words list
print(stopwords.words("english"))
clean_text = [x for x in words if x not in stopwords.words("english")]
clean_text = " ".join(clean_text)
print("\n")
print(clean_text)


# In[54]:


first3000 = data["content"][0:3000]


# In[57]:


stop = stopwords.words("english")
def cleanText(text):
    cleaned = BeautifulSoup(text, "lxml")
    cleaned = re.sub("[^a-zA-Z]", " ", cleaned.get_text())
    words = cleaned.split()
    words = [word for word in words if word not in stop]
    words = " ".join(words)
    return(words)

allText = [cleanText(text) for text in data["content"][0:3000]]


# In[60]:


(train_contents, test_contents, train_target, test_target) = train_test_split(allText, data["score"][0:3000],test_size = 0.2)

#Bag of Words with 5000 most common words
vectorizer = CountVectorizer(analyzer='word',max_features = 5000)
vectorizer.fit(train_contents)
train_word_columns = vectorizer.transform(train_contents).toarray()
test_word_columns = vectorizer.transform(test_contents).toarray()
print(train_word_columns)


# In[61]:


#Multinomial Naive Bayes 
mnb = MultinomialNB()
mnb.fit(train_word_columns,train_target)
preds = mnb.predict(test_word_columns)
print(accuracy_score(preds,test_target))


# In[62]:


#Random Forest Classifier
rfc = RandomForestClassifier(n_estimators = 1000)#parameter
rfc.fit(train_word_columns,train_target)
preds = rfc.predict(test_word_columns)
print(accuracy_score(preds,test_target))


# In[63]:


#Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(train_word_columns,train_target)
preds = gnb.predict(test_word_columns)
print(accuracy_score(preds,test_target))


# In[64]:


#Support Vector Classifier
svt = SVC(kernel="linear")
svt.fit(train_word_columns,train_target)

preds = svt.predict(test_word_columns)

svm_acc = accuracy_score(preds,test_target)
print(svm_acc)


# In[65]:


#My 2nd Support Vector Classifier
svt = SVC(kernel = "poly")
svt.fit(train_word_columns,train_target)

preds = svt.predict(test_word_columns)

svm_acc = accuracy_score(preds,test_target)
print(svm_acc)


# In[74]:


#kNN Classifier Algorithm
kVals=[3, 5, 15, 30, 60, 120, 250]
kAcc=[]
def KnnPrediction(k):
    MykNN = KNeighborsClassifier(n_neighbors=k, weights='distance')
    MykNN.fit(train_word_columns,train_target)
    preds = MykNN.predict(test_word_columns)
    MykNN_acc = accuracy_score(preds,test_target)
    return (MykNN_acc)

for k in kVals:
    kAcc.append(KnnPrediction(k))
    
plt.suptitle('KNN algorithm',fontsize=14)
plt.xlabel('k neighbors')
plt.ylabel('Accuracy')
plt.plot(kVals,kAcc,'ro-',label='k-NN')
plt.legend(loc='lower left', shadow=True)
plt.axis([0,max(kVals),0,1])
plt.show()
max(kAcc)


# In[67]:


extractor = PCA(n_components=270, whiten = True)  # can change the number of components, doesn't work without whiten=true

extractor.fit(train_word_columns)

transformed_train_data = extractor.transform(train_word_columns)
#transformed_train_data = abs(train_word_columns)
transformed_test_data = extractor.transform(test_word_columns)
#transformed_test_data = abs(test_word_columns)
print(transformed_train_data)
print(transformed_train_data[2])


# In[68]:


#Support Vector Classifier 
svt = SVC(kernel="linear")
svt.fit(transformed_train_data,train_target)

preds = svt.predict(transformed_test_data)

svm_acc = accuracy_score(preds,test_target)
print(svm_acc)


# In[69]:


#Support Vector Classifier 2
svt = SVC(kernel="poly")
svt.fit(transformed_train_data,train_target)

preds = svt.predict(transformed_test_data)

svm_acc = accuracy_score(preds,test_target)
print(svm_acc)


# In[70]:


#Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(transformed_train_data,train_target)
preds = gnb.predict(transformed_test_data)
print(accuracy_score(preds,test_target))


# In[71]:


#Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(abs(transformed_train_data),train_target)
preds = mnb.predict(abs(transformed_test_data))
print(accuracy_score(preds,test_target))


# In[72]:


#Random Forest Classifier
rfc = RandomForestClassifier(n_estimators = 1000)#parameter
rfc.fit(transformed_train_data,train_target)
preds = rfc.predict(transformed_test_data)
print(accuracy_score(preds,test_target))


# In[73]:


#kNN Classifier Algorithm
kVals=[3, 5, 15, 30, 60, 120, 250]
kAcc=[]
def KnnPrediction(k):
    MykNN = KNeighborsClassifier(n_neighbors=k, weights='distance')
    MykNN.fit(transformed_train_data,train_target)
    preds = MykNN.predict(transformed_test_data)
    MykNN_acc = accuracy_score(preds,test_target)
    return (MykNN_acc)

for k in kVals:
    kAcc.append(KnnPrediction(k))
    
plt.suptitle('KNN algorithm',fontsize=14)
plt.xlabel('k neighbors')
plt.ylabel('Accuracy')
plt.plot(kVals,kAcc,'ro-',label='k-NN')
plt.legend(loc='lower left', shadow=True)
plt.axis([0,max(kVals),0,1])
plt.show()
max(kAcc)


# In[ ]:





# In[ ]:




