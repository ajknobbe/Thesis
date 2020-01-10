#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## decision tree 2012


# Load libraries
import os
os.chdir('c:/Users\\aknob\Desktop\Data')

import numpy as np
import pandas as pd
import sklearn

import matplotlib.pyplot as plt 
plt.rc("font", size=14)

from sklearn import preprocessing
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.experimental import enable_iterative_imputer ## impute nan values
from sklearn.impute import IterativeImputer
from sklearn import utils
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score


#silc for Statistics on Income and Living Conditions
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

## final +  cleaned dataframe is silc

# select x and y 
X = silcSubSet.drop(['PH010'], axis = 1)
y = silcSubSet[['PH010']]
# PH010 is Self-perceived general health

## import additional packages for Random Forest
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

# Build a forest and compute the feature importances

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Split the data into 20% test and 80% training 
## should the occur before imputation?
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the classifier
clf.fit(X_train, y_train)

# Print the name and gini importance of each feature
print("Feature importance per variable")
for feature in zip(silcSubSet.columns, clf.feature_importances_):
    print(feature)
    
    
# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.025    
sfm = SelectFromModel(clf, threshold=0.02)

# Train the selector
sfm.fit(X_train, y_train)

# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(silcSubSet.columns[feature_list_index])
                      
# Train the selector
sfm.fit(X_train, y_train)

# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(silcSubSet.columns[feature_list_index])
    
# Transform the data to create a new dataset containing only the most important features
# Note: We have to apply the transform to both the training X and test X data.
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)

# Create a new random forest classifier for the most important features
clf_important = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the new classifier on the new dataset containing the most important features
clf_important.fit(X_important_train, y_train)


# Apply The Full Featured Classifier To The Test Data
y_pred = clf.predict(X_test)

# View The Accuracy Of Our Full Feature (4 Features) Model
print('Accuracy score full feature:')
print(accuracy_score(y_test, y_pred))


# Apply The Full Featured Classifier To The Test Data
y_important_pred = clf_important.predict(X_important_test)

# View The Accuracy Of Our Limited Feature (2 Features) Model
print('Accuracy score limited feature:')
print(accuracy_score(y_test, y_important_pred))


# In[ ]:





# In[15]:


silcSubSet['PH010'].value_counts()

silcSubSet = silc.dropna(axis=1, thresh = 4000) # drop columns that have > 4000 NaN
silcSubSet = silcSubSet.dropna(axis=0, thresh = 120) # drop rows that have less than 120 filled cells
## these values make sure that each row and collumn is filled enough to properly use imputation

silcSubSet = silcSubSet.drop(['HB020', 'PB020', 'RB020', 'DB020', "PB220A", 'PL111'], axis = 1) # all values are NL, as it is dutch data, and thus not influenctal
## scores with 020 are country, pb220a is country of birth, PL111 is working sector categorized by letters

## imputate the NaN values that are left by adding the mean of that column 
silcSubSet = silcSubSet.fillna(silcSubSet.mean(), inplace=False)

## some categories (e.g. continuous) of variables are changed and now usable by sk learn
silcSubSet = silcSubSet.apply(LabelEncoder().fit_transform) 


# In[16]:


silc['PH010'].value_counts()


# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


print(np.unique(silcSubSet['PH010'].values))

plt.hist(silcSubSet['PH010'])
plt.ylabel('Frequency')
plt.xlabel('Self Perceived General Health')
plt.title('Histogram')

## add to thesis?


# In[ ]:


## decision tree 2013
## calculate feature importance, however, what feature is what code?

# Load libraries
import os
os.chdir('c:/Users\\aknob\Desktop\Data')

import numpy as np
import pandas as pd
import sklearn

import matplotlib.pyplot as plt 
plt.rc("font", size=14)

from sklearn import preprocessing
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.experimental import enable_iterative_imputer ## impute nan values
from sklearn.impute import IterativeImputer # imputation
from sklearn import utils
from sklearn.preprocessing import LabelEncoder #transform categories to be usable by sklearn
from sklearn.feature_selection import SelectFromModel #allows to select most important features
from sklearn.metrics import accuracy_score #print model accuracy

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

## final +  cleaned dataframe is silc

# select x and y 
X = silcSubSet.drop(['PH010'], axis = 1)
y = silcSubSet[['PH010']]
# PH010 is Self-perceived general health

## import additional packages for Random Forest
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Build a forest and compute the feature importances

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Split the data into 20% test and 80% training 
## should the occur before imputation?
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the classifier
clf.fit(X_train, y_train)

# Print the name and gini importance of each feature
print("Feature importance per variable")
for feature in zip(silcSubSet.columns, clf.feature_importances_):
    print(feature)
    
    
# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.02    
sfm = SelectFromModel(clf, threshold=0.02)

# Train the selector
sfm.fit(X_train, y_train)

# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(silcSubSet.columns[feature_list_index])
                      
# Train the selector
sfm.fit(X_train, y_train)

# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(silcSubSet.columns[feature_list_index])
    
# Transform the data to create a new dataset containing only the most important features
# Note: We have to apply the transform to both the training X and test X data.
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)

# Create a new random forest classifier for the most important features
clf_important = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the new classifier on the new dataset containing the most important features
clf_important.fit(X_important_train, y_train)

# Apply The Full Featured Classifier To The Test Data
y_pred = clf.predict(X_test)

# View The Accuracy Of Our Full Feature (4 Features) Model
print('Accuracy score full feature:')
print(accuracy_score(y_test, y_pred))

# Apply The Full Featured Classifier To The Test Data
y_important_pred = clf_important.predict(X_important_test)

# View The Accuracy Of Our Limited Feature (2 Features) Model
print('Accuracy score limited feature:')
print(accuracy_score(y_test, y_important_pred))

