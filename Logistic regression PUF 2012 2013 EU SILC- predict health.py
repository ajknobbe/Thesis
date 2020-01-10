#!/usr/bin/env python
# coding: utf-8

# In[2]:


## logistic regression for 2012
import os
os.chdir('c:/Users\\aknob\Desktop\Data')

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns


silc = pd.read_csv('thesis_data_2012.csv')

silcSubSet = silc.dropna(axis=1, thresh = 4000) # drop columns that have > 4000 NaN
silcSubSet = silcSubSet.dropna(axis=0, thresh = 120) # drop rows that have  NaN

silcSubSet = silcSubSet.drop(['HB020', 'PB020', 'RB020', 'DB020', "PB220A", 'PL111'], axis = 1) # all values are NL, as it is dutch data, and thus not influenctal
## scores with 020 are country, pb220a is country of birth, PL111 is working sector categorized by letters

## imputate the NaN values that are left by adding the mean of that column 
silcSubSet = silcSubSet.fillna(silcSubSet.mean(), inplace=False)

## some categories (e.g. continuous) of variables are changed and now usable by sk learn
silcSubSet = silcSubSet.apply(LabelEncoder().fit_transform) 



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

# import data 
#features with more than 2% influence from random forest, based on 2012! used to predict 2013
x = silcSubSet[['HY010','HY020', 'HY022', 'HY023','HY090G','HY140G', 'HS130', 'HH070', 'PB060', 'PB140', 'PL051', 'PH060', 'PY021G','RB050']]
# 26% accuracy swith subset x


y = silcSubSet[['PH010']]

# split X and y into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1) ##75% used for training 25% for test

# Create an instance of Logistic Regression Classifier and fit the data.

lr = LogisticRegression(class_weight = 'balanced')
train_test_split(x, y)
lr.fit(x_train, y_train)
lr.predict(x_test)
y_pred = lr.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# test score --> why do we have no false p/n but only 96% accuracy?
score = lr.score(x_test, y_test)
print(score)

#predict probability
#print(lr.predict_proba(x_test))

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix,annot=True,cbar=True) ##this needs tweaking



# In[23]:


import os
os.chdir('c:/Users\\aknob\Desktop\Data')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel


silc = pd.read_csv('thesis_data_2013.csv')

silcSubSet = silc.dropna(axis=1, thresh = 4000) # drop columns that have > 4000 NaN
silcSubSet = silcSubSet.dropna(axis=0, thresh = 120) # drop rows that have  NaN


silcSubSet = silcSubSet.drop(['HB020', 'PB020', 'RB020', 'DB020', "PB220A", 'PL111'], axis = 1) # all values are NL, as it is dutch data, and thus not influenctal
## scores with 020 are country, pb220a is country of birth, PL111 is working sector categorized by letters

## imputate the NaN values that are left by adding the mean of that column 
silcSubSet = silcSubSet.fillna(silcSubSet.mean(), inplace=False)

## some categories (e.g. continuous) of variables are changed and now usable by sk learn
silcSubSet = silcSubSet.apply(LabelEncoder().fit_transform) 

y = silcSubSet[['HY010']]


# In[34]:


print(y.min())
print(y.max())
print(y.mean())
print(y.median())

ybin = silcSubSet['HY010']
ybin = ybin  > 2923
print(sum(ybin == True),sum(ybin == False),len(ybin))
#if mmore than 2500 1, otherwise 2


# In[1]:


## logistic regression for 2013
import os
os.chdir('c:/Users\\aknob\Desktop\Data')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel


silc = pd.read_csv('thesis_data_2013.csv')

silcSubSet = silc.dropna(axis=1, thresh = 4000) # drop columns that have > 4000 NaN
silcSubSet = silcSubSet.dropna(axis=0, thresh = 120) # drop rows that have  NaN


silcSubSet = silcSubSet.drop(['HB020', 'PB020', 'RB020', 'DB020', "PB220A", 'PL111'], axis = 1) # all values are NL, as it is dutch data, and thus not influenctal
## scores with 020 are country, pb220a is country of birth, PL111 is working sector categorized by letters

## imputate the NaN values that are left by adding the mean of that column 
silcSubSet = silcSubSet.fillna(silcSubSet.mean(), inplace=False)

## some categories (e.g. continuous) of variables are changed and now usable by sk learn
silcSubSet = silcSubSet.apply(LabelEncoder().fit_transform) 

## PH010 = self percieved general health, H
## HS021 = arrears on utility bills
## HS011 = arreears on rent/mortgage
## HS040 = ability to go on holY010 = Total household gross income, iday for atleast 1 week
## HS050 = ability to include proteins in mean atleast every other day
## HS060 = Capacity to face unexpected financial expenses
## HS120 = Ability to make ends meet, 
## HH070 = total housing cost, 
## HH050 = ability to keep home (financially) adequately warm (yes/no)
## HY010 = total household gross income

# import data 

#manual prediction x
#x = silcSubSet[['PH010', 'HS021', 'HS011', 'HS040', 'HS050', 'HS060', 'HS120', 'HY010','HH070']]

#features with more than 2% influence from random forest, based on 2012! used to predict 2013
x = silcSubSet[['HY010','HY020', 'HY022', 'HY023','HY090G','HY140G', 'HS130', 'HH070', 'PB060', 'PB140', 'PL051', 'PH060', 'PY021G','RB050']]
# 26% accuracy swith subset x


y = silcSubSet[['PH010']]

# split X and y into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1) ##75% used for training 25% for test

# Create an instance of Logistic Regression Classifier and fit the data.

lr = LogisticRegression(class_weight = 'balanced')
train_test_split(x, y)
lr.fit(x_train, y_train)
lr.predict(x_test)
y_pred = lr.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# test score --> why do we have no false p/n but only 96% accuracy?
score = lr.score(x_test, y_test)
print(score)

#predict probability
#print(lr.predict_proba(x_test))

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



#sns.heatmap(confusion_matrix,annot=True,cbar=True) ##too large numbers?



# In[ ]:




