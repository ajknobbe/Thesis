#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[ ]:





# In[13]:


## Decision tree 2012

# Load libraries
import os
os.chdir('c:/Users\\aknob\Desktop\Data')

import numpy as np
import pandas as pd
import sklearn

import matplotlib.pyplot as plt 
plt.rc("font", size=14)

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel

silc = pd.read_csv('thesis_data_2012.csv')

silcSubSet = silc.dropna(axis=1, thresh = 4000) # drop columns that have > 4000 NaN
silcSubSet = silcSubSet.dropna(axis=0, thresh = 120) # drop rows that have less than 120 filled cells
## these values make sure that each row and collumn is filled enough to properly use imputation

silcSubSet = silcSubSet.drop(['HB020', 'PB020', 'RB020', 'DB020', "PB220A", 'PL111'], axis = 1) # all values are NL, as it is dutch data, and thus not influenctal
## scores with 020 are country, pb220a is country of birth, PL111 is working sector categorized by letters

## imputate the NaN values that are left by adding the mean of that column 
silcSubSet = silcSubSet.fillna(silcSubSet.mean(), inplace=False)

## some categories (e.g. continuous) of variables are changed and now usable by sk learn
silcSubSet = silcSubSet.apply(LabelEncoder().fit_transform) 

# import data 

#manual x, use/blur one x
## x with predictors greater than 2% via random forest, based on 2012
x = silcSubSet[['HY010','HY020', 'HY022', 'HY023','HY090G','HY140G', 'HS130', 'HH070', 'PB060', 'PB140', 'PL051', 'PH060', 'PY021G','RB050']]
y = silcSubSet[['PH010']] 

# 'HC020' is dropped for x, since it didnt occur in the 2013 set


## PH010 = self percieved general health, HY010 = Total household gross income, 
## HS021 = arrears on utility bills
## HS011 = arreears on rent/mortgage
## HS040 = ability to go on holiday for atleast 1 week
## HS050 = ability to include proteins in mean atleast every other day
## HS060 = Capacity to face unexpected financial expenses
## HS120 = Ability to make ends meet, 
## HH070 = total housing cost, 
## HH050 = ability to keep home (financially) adequately warm (yes/no)
## HY010 = total household gross income

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1) ##80% used for training 25% for test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(class_weight = 'balanced')

# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

# Model Accuracy, how often is the classifier correct?

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))





# In[ ]:





# In[14]:


## Decision tree 2013

# Load libraries
import os
os.chdir('c:/Users\\aknob\Desktop\Data')

import numpy as np
import pandas as pd
import sklearn

import matplotlib.pyplot as plt 
plt.rc("font", size=14)

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel

silc = pd.read_csv('thesis_data_2013.csv')

silcSubSet = silc.dropna(axis=1, thresh = 4000) # drop columns that have > 4000 NaN
silcSubSet = silcSubSet.dropna(axis=0, thresh = 120) # drop rows that have less than 120 filled cells
## these values make sure that each row and collumn is filled enough to properly use imputation

silcSubSet = silcSubSet.drop(['HB020', 'PB020', 'RB020', 'DB020', "PB220A", 'PL111'], axis = 1) # all values are NL, as it is dutch data, and thus not influenctal
## scores with 020 are country, pb220a is country of birth, PL111 is working sector categorized by letters

## imputate the NaN values that are left by adding the mean of that column 
silcSubSet = silcSubSet.fillna(silcSubSet.mean(), inplace=False)

## some categories (e.g. continuous) of variables are changed and now usable by sk learn
silcSubSet = silcSubSet.apply(LabelEncoder().fit_transform) 

# import data 

#manual prediction
#x = silcSubSet[['PH010', 'HS021', 'HS011', 'HS040', 'HS050', 'HS060', 'HS120', 'HY010','HH070']]

#features with more than 2% influence from random forest, based on 2012! used to predict 2013
x = silcSubSet[['HY010','HY020', 'HY022', 'HY023','HY090G','HY140G', 'HS130', 'HH070', 'PB060', 'PB140', 'PL051', 'PH060', 'PY021G','RB050']]
# 26% accuracy swith subset x


y = silcSubSet[['PH010']]

## PH010 = self percieved general health, HY010 = Total household gross income, 
## HS021 = arrears on utility bills
## HS011 = arreears on rent/mortgage
## HS040 = ability to go on holiday for atleast 1 week
## HS050 = ability to include proteins in mean atleast every other day
## HS060 = Capacity to face unexpected financial expenses
## HS120 = Ability to make ends meet, 
## HH070 = total housing cost, 
## HH050 = ability to keep home (financially) adequately warm (yes/no)
## HY010 = total household gross income

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1) ##75% used for training 25% for test


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(class_weight = 'balanced')

# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

# Model Accuracy, how often is the classifier correct?

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

## package graphiz wont load https://www.datacamp.com/community/tutorials/decision-tree-classification-python
## https://scikit-learn.org/stable/modules/tree.html

## after importing numpy and pandas fail to work --> reinstall needed

#from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO  
#from IPython.display import Image  
#import pydotplus

#dot_data = StringIO()
#export_graphviz(clf, out_file=dot_data,  
#                filled=True, rounded=True,
#                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('diabetes.png')
#Image(graph.create_png())


# In[4]:


from sklearn.model_selection import ShuffleSplitn_samples = silcSubSet.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
cross_val_score(clf, silcSubSet, iris.target, cv=cv)  

