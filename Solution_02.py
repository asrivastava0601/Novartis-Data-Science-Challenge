#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 21:30:25 2020

@author: abhinavsrivastava
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Train.csv")


data["DATE"].value_counts()


data.isnull().values.any()

data.isnull().sum()

df = data.drop(['INCIDENT_ID','DATE'], axis = 1)


df['MULTIPLE_OFFENSE'].value_counts()


stats = df.describe()

st = df.apply(pd.Series.value_counts)

df.isnull().values.any()

df.isnull().sum()


df = df.fillna(value = 1)

df.isnull().sum()

print(df.shape)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Univariate Analysis ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif

X = df.iloc[:,0:15] #independent columns
y = df.iloc[:,15:]


############# f_classif ##########

bestfeatures = SelectKBest(score_func= f_classif, k=10)
fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns

featureScores

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



X_data = X[['X_8','X_10','X_11','X_12','X_15']]



from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report 
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import f1_score


X_train, X_test, y_train, y_test = train_test_split( X_data, y, test_size = 0.2, random_state = 42)



######## Decision Tree model ########
classifier_dt = DecisionTreeClassifier(criterion = "entropy", max_depth = 30, min_samples_leaf = 5)

fitting_dt = classifier_dt.fit(X_train,y_train)

y_pred_dt = classifier_dt.predict(X_test)

accuracy_dt = accuracy_score(y_test ,y_pred_dt) *100


print("\nAccuracy Decision tree - Training set: ", classifier_dt.score(X_train,y_train)*100)
print ("\nAccuracy Decision tree - Testing Tree: ",  accuracy_dt)

print("Confusion Matrix: ", confusion_matrix(y_test, y_pred_dt)) 
  
print("Report : \n", classification_report(y_test, y_pred_dt))

disp_dt = plot_confusion_matrix(fitting_dt, X_test, y_test,cmap=plt.cm.Blues, values_format = '.4g')



########### Precision-Recall Curve ############
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


learning_probability = fitting_dt.predict_proba(X_test)

# keep probabilities for the positive outcome only
learning_probability = learning_probability[:, 1]

# predict class values
y_hat = fitting_dt.predict(X_test)

learning_precision, learning_recall, _ = precision_recall_curve(y_test, learning_probability)

learning_f1, learning_auc = f1_score(y_test, y_hat), auc(learning_recall, learning_precision)

# F1-score and AUC
print('\nDecision Tree- : f1=%.3f auc=%.3f' % (learning_f1, learning_auc))

plt.plot(learning_recall, learning_precision, marker='.', label='Decision Tree')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("Precision-Recall Curve")

plt.legend()

plt.show()




######### Predict for the test set given #########

test_df = pd.read_csv("Test.csv")

new_test = test_df.drop(["INCIDENT_ID", "DATE"], axis =1)

new_test.isnull().values.any()

new_test.isnull().sum()

new_test = new_test.fillna(value = 1)

new_test.isnull().sum()

print(new_test.shape)


X_testset = new_test[['X_8','X_10','X_11','X_12','X_15']]

y_pred_testset = pd.DataFrame(fitting_dt.predict(X_testset))

y_pred_testset[0].value_counts()

y_pred_testset.rename(columns= {0: "MULTIPLE_OFFENSE"}, inplace= True)

test_df = pd.concat([test_df,y_pred_testset], axis =1)



#### Print the results in csv format ####

transfer = ['INCIDENT_ID','MULTIPLE_OFFENSE']

test_df.loc[:,transfer].to_csv("Solution01.csv", index = False, header= True)

