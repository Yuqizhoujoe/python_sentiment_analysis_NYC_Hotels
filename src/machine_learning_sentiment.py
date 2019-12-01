## Import nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')


## Import models and evaluation functions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, train_test_split


## Import vectorizers to turn text into numeric
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 

## Import common module
import re
import string
import csv
from string import punctuation
import os

## Import pandas to read in data
import pandas as pd
import numpy as np

# import plotting
from matplotlib import pyplot as plt

""" Data Preparation """
data = pd.read_csv("data.txt", delimiter="\t", header=None)
data.columns = ["rating", "reviews"]
data['rating'].hist()
data['rating'].value_counts()

# 9576 records and 2 columns: reviews and rates
data.shape

# drop rate-30 class
data['rating'] = data['rating'].apply(lambda x: np.nan if x == 30 else int(x))  
data = data.dropna()

""" unused code 
# drop all the rates in 10 && 30
# data_10 = data[data['rating'] == 10]
# data_10 = data_10.sample(n = 500)

# append rates 10 dataframe with reduced numbers 
# data = data.append(data_10)
# data['rating'].value_counts()
"""

# convert numeric rates to positive and negative 
data["rating"] = data.loc[:]["rating"].apply(lambda x: 1 if x>30 else 0)
data

X_text = data['reviews']
Y = data['rating']

""" Countvectorizer """
# create a tfidvectorizer
count_vectorizer = CountVectorizer(ngram_range=(1,3))

# Let the vectorizer learn what tokens exist in the text data
count_vectorizer.fit(X_text)

# Turn these tokens into a numeric matrix
Xc = count_vectorizer.transform(X_text)

""" Tfidvectorizer """
# create a tfidvectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3))

# Let the vectorizer learn what tokens exist in the text data
tfidf_vectorizer.fit(X_text)

# Turn these tokens into a numeric matrix
Xt = tfidf_vectorizer.transform(X_text)

## train test split
# Tridvectorizer
X_train, X_test, y_train, y_test = train_test_split(Xt, Y, train_size=0.7)
# Countvectorizer
X_trainc, X_testc, y_trainc, y_testc = train_test_split(Xc, Y, train_size=0.7)

""" Modeling Svm """
# try different inverse regularization parameters
C = [0.1,1,10,100,1000]
# create empty list to save all the results
tf_accs_svm = []
cv_accs_svm = []
for c in C:
    # create parameter grid for specific c
    param_grid_svm = {
        'C': [c]
    } 
    # create empty unlearned svm model
    svm = LinearSVC()
    # create grid to get 10-fold cross validation and contain the parameter grid 
    grid_svm = GridSearchCV(svm, param_grid=param_grid_svm, cv=10)
    grid_svmc = GridSearchCV(svm, param_grid=param_grid_svm, cv=10)
    # fit and train the random forest
    grid_svm.fit(X_train,y_train)
    grid_svmc.fit(X_trainc, y_trainc)
    # get cv results and keep tracking them
    tf_accs_svm.append(grid_svm.best_score_)
    cv_accs_svm.append(grid_svmc.best_score_)

# plotting the results
plt.plot(C,tf_accs_svm)
plt.plot(C,cv_accs_svm)
plt.show()

""" Modeling Naive Bayes """
# try different alphas
alphas = np.linspace(0,1,5)
# create empty list to save all the results
tf_accs_mnb = []
cv_accs_mnb = []
for alpha in alphas:
    # create parameter grid for specified alpha
    param_mnb = {
        'alpha': [alpha]
    }
    # create model - MultinomialNB
    mnb = MultinomialNB()
    # create grid to get 10-fold cross validation and contain the parameter grid 
    grid_mnb = GridSearchCV(mnb, param_grid=param_mnb,cv=10)
    grid_mnbc = GridSearchCV(mnb, param_grid=param_mnb,cv=10)
    # fit and train the Multinomial Naive Bayes  
    grid_mnb.fit(X_train, y_train)
    grid_mnbc.fit(X_trainc, y_trainc)
    # keep tracking of the results
    tf_accs_mnb.append(grid_mnb.best_score_)
    cv_accs_mnb.append(grid_mnbc.best_score_)

# plotting the results
plt.plot(alphas,tf_accs_mnb)
plt.plot(alphas,cv_accs_mnb)
plt.show()

""" Modeling Logistic Regression """
# try different inverse regularization parameters
C = [0.1,1,10,100,1000]
tf_accs_lr = []
cv_accs_lr = []
for c in C:
    # create parameter grid for specified c
    param_grid_svm = {
        'C': [c]
    } 
    # create empty unlearned svm model
    lr = LogisticRegression()
    # create grid to get 10-fold cross validation and contain the parameter grid 
    grid_lr = GridSearchCV(lr, param_grid=param_grid_svm, cv=10)
    grid_lrc = GridSearchCV(lr, param_grid=param_grid_svm, cv=10)
    # fit and train the logistic regression
    grid_lr.fit(X_train,y_train)
    grid_lrc.fit(X_trainc,y_trainc)
    # get cv results and keep tracking them
    tf_accs_lr.append(grid_lr.best_score_)
    cv_accs_lr.append(grid_lrc.best_score_)

# plotting the results
plt.plot(C,tf_accs_lr)
plt.plot(C,cv_accs_lr)
plt.show()

## compare results
# create list to save all the results
accuracy = {'TF-SVM':tf_accs_svm, 
            'TF-MNB':tf_accs_mnb,
            'TF-LR':tf_accs_lr, 
            'CV-SVM':cv_accs_svm,
            'CV-MNB':cv_accs_mnb,
            'CV-LR':cv_accs_lr}
# sort out the results DESC
for key in accuracy:
    accuracy[key].sort(reverse=True)
# create dataframe for all the results 
accs_result = pd.DataFrame(accuracy)
accs_result

## model evaluation
print('Accuracy of Linear SVM: %s' %(accuracy_score(y_test, grid_svm.predict(X_test))))
print('Accuracy of Linear SVMc: %s' %(accuracy_score(y_testc, grid_svmc.predict(X_testc))))
print('Accuracy of Naive Bayes: %s' %(accuracy_score(y_test, grid_mnb.predict(X_test))))
print('Accuracy of Naive Bayesc: %s' %(accuracy_score(y_testc, grid_mnbc.predict(X_testc))))
print('Accuracy of Logistic Regression: %s' %(accuracy_score(y_test, grid_lr.predict(X_test))))
print('Accuracy of Logistic Regressionc: %s' %(accuracy_score(y_testc, grid_lrc.predict(X_testc))))

""" 
After comparing the results within three models, SVM, Naive Bayes, Logistic Regression,
We choose Model SVM which uses count vectorizer as our final model 
""" 

""" unused code 
# cvs = range(5,30,5)
# tf_accs_svm = []
# cv_accs_svm = []
# for cv in cvs:
#     # create grid for model tuning
#     param_grid = {
#         'C': [grid_svm.best_params_['C']]
#     } 
#     # create empty unlearned svm model
#     svm = LinearSVC()
#     # tune the model and get 10-fold cross validation results 
#     best_svm = GridSearchCV(svm, param_grid=param_grid_svm, cv=cv)
#     best_svmc = GridSearchCV(svm, param_grid=param_grid_svm, cv=cv)
#     # fit and train the random forest
#     best_svm.fit(X_train,y_train)
#     best_svmc.fit(X_trainc, y_trainc)
#     # get cv results and keep tracking them
#     tf_accs_svm.append(best_svm.best_score_)
#     cv_accs_svm.append(best_svmc.best_score_)

# # plotting the results
# plt.plot(cvs,tf_accs_svm)
# plt.plot(cvs,cv_accs_svm)
# plt.show()

# tf_accs_svm = []
# cv_accs_svm = []
# # create grid for model tuning
# param_grid = {
#     'C': [grid_svm.best_params_['C']]
# } 
# # create empty unlearned svm model
# svm = LinearSVC()
# # tune the model and get 10-fold cross validation results 
# final_svm = GridSearchCV(svm, param_grid=param_grid_svm, cv=20)
# final_svmc = GridSearchCV(svm, param_grid=param_grid_svm, cv=20)
# # fit and train the random forest
# final_svm.fit(X_train,y_train)
# final_svmc.fit(X_trainc, y_trainc)
# # get cv results and keep tracking them
# tf_accs_svm.append(final_svm.best_score_)
# cv_accs_svm.append(final_svmc.best_score_)

# print('Accuracy of Linear SVM: %s' %(accuracy_score(y_test, final_svm.predict(X_test))))
"""


# use dictionary to store the vectorizer's feature names and their corresponding coefficients
feature_to_coef = {
    word: coef for word, coef in zip(
        count_vectorizer.get_feature_names(), grid_svmc.best_estimator_.coef_[0]
    )
}

# get the positive words && phrases
with open('pos.txt', 'w') as f:
    for best_positive in sorted(
        feature_to_coef.items(), 
        key=lambda x: x[1], 
        reverse=True)[:100]:
        f.write(best_positive[0] + '\t' + str(best_positive[1]) + '\n')

# get the negative words and phrases
with open('neg.txt', 'w') as f:
    for best_negative in sorted(
        feature_to_coef.items(), 
        key=lambda x: x[1])[:100]:
        f.write(best_negative[0] + '\t' + str(best_negative[1]) + '\n')

""" 
"Maria" is the best positive words from the model
"Maria" is a front desk worker in penn hotel
"""

""" unused code
# with open('neg.txt', 'w') as f:
#     for best_negative in sorted(
#         feature_to_coef.items(), 
#         key=lambda x: x[1], 
#         reverse=True)[:50]:
#         f.write(best_negative[0] + '\t' + str(best_negative[1]) + '\n')
        
# with open ('pos.txt', 'w') as f:
#     for best_positive in sorted(
#         feature_to_coef.items(), 
#         key=lambda x: x[1])[:50]:
#         f.write(best_positive[0] + '\t' + str(best_positive[1]) + '\n')
 """


