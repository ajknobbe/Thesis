#!/usr/bin/env python
# coding: utf-8

# In[2]:


#set working directory

import os
os.getcwd()
os.chdir('c:/Users\\aknob\Desktop\Data')


# In[3]:


#import packages for this notebook

import pandas as pd
import numpy
import os
import glob


# In[134]:


#code for 2012 data
#huishouden means household

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

# H
f1 = pd.read_csv('NL_2012h_EUSILC.csv')
f1.HB030 = f1["HB030"].apply(str)
#print(f1.head())

# P
f2 = pd.read_csv('NL_2012p_EUSILC.csv')
f2.PB030 = f2["PB030"].apply(str)
f2["HuishoudenP"] = f2.PB030.str[:-2]
#print(f2.Huishouden.head())
#print(f2.PB030.head())

# H only has variable huishouden, therefore merge on huishouden
combined = f1.merge(f2, left_on="HB030", right_on="HuishoudenP", how="outer")
#print(combined.head())

# R
f3 = pd.read_csv('NL_2012r_EUSILC.csv')
f3.RB030 = f3["RB030"].apply(str)
f3["HuishoudenR"] = f3.RB030.str[:-2]
#print(f3.Huishouden.head())
#print(f3.head())

# P and R both have a personal number
combined = combined.merge(f3, left_on="PB030", right_on="RB030", how="outer")
#print(combined.head())

# D
f4 = pd.read_csv('NL_2012d_EUSILC.csv')
f4.DB030 = f4["DB030"].apply(str)
#f4["HuishoudenD"] = f4.DB030.str[:-2]
print(f4["DB030"].head())

# D has a personal number
combined = combined.merge(f4, left_on="HB030", right_on="DB030", how="outer")
#print(combined.head())

#print(combined[["HB030", "PB030", "DB030", "RB030", "HuishoudenP"]].head())
#how much of the data from different categories can be matched?
print(sum(combined["HB030"] == combined["HuishoudenP"])/len(combined))
print(sum(combined["PB030"] == combined["RB030"])/len(combined))
print(sum(combined["HB030"] == combined["DB030"])/len(combined))


# In[135]:


#only run after running code for data  2012! Preferably just once at all

#combined.to_csv( "thesis_data_2012.csv", index=False, encoding='utf-8-sig')


# In[4]:


import numpy as np
import pandas as pd
f1 = pd.read_csv('thesis_data_2012.csv')
countNan = f1.isna().sum()
print(len(np.asarray(countNan)))
lessNan = np.asarray(countNan) > 4000
idx_to_keep= np.where(lessNan)[0]
print(len(idx_to_keep))
#df.drop(df.columns[i], axis=1)
f2 = f1.drop(f1.columns[idx_to_keep],axis=1)
countNan = f2.isna().sum()
print(len(np.asarray(countNan)))


# In[26]:


import numpy as np
import pandas as pd
f1 = pd.read_csv('NL_2013d_EUSILC.csv')
countNan = f1.isna().sum()
print(len(np.asarray(countNan)))
lessNan = np.asarray(countNan) > 400
idx_to_keep= np.where(lessNan)[0]
print(len(idx_to_keep))
#df.drop(df.columns[i], axis=1)
f2 = f1.drop(f1.columns[idx_to_keep],axis=1)
countNan = f2.isna().sum()
print(len(np.asarray(countNan)))


# In[8]:


# code for 2013 data

# H
f1 = pd.read_csv('NL_2013h_EUSILC.csv')
f1.HB030 = f1["HB030"].apply(str)
#print(f1.head())

# P
f2 = pd.read_csv('NL_2013p_EUSILC.csv')
f2.PB030 = f2["PB030"].apply(str)
f2["HuishoudenP"] = f2.PB030.str[:-2]
#print(f2.Huishouden.head())
#print(f2.PB030.head())

# H only has variable 'huishouden',therefore merge on huishouden
combined = f1.merge(f2, left_on="HB030", right_on="HuishoudenP", how="outer")
#print(combined.head())

# R
f3 = pd.read_csv('NL_2013r_EUSILC.csv')
f3.RB030 = f3["RB030"].apply(str)
f3["HuishoudenR"] = f3.RB030.str[:-2]
#print(f3.Huishouden.head())
#print(f3.head())

# P and R both have a personal number
combined = combined.merge(f3, left_on="PB030", right_on="RB030", how="outer")
#print(combined.head())

# D
f4 = pd.read_csv('NL_2013d_EUSILC.csv')
f4.DB030 = f4["DB030"].apply(str)
#f4["HuishoudenD"] = f4.DB030.str[:-2]
#print(f4["DB030"].head())

# D has a personal number
combined = combined.merge(f4, left_on="HB030", right_on="DB030", how="outer")
print(combined.shape)

#how much of the data from different categories can be matched?
print(sum(combined["HB030"] == combined["HuishoudenP"])/len(combined))
print(sum(combined["PB030"] == combined["RB030"])/len(combined))
print(sum(combined["HB030"] == combined["DB030"])/len(combined))


# In[137]:


#only run after running code for data  2013! Preferably just once at all

#combined.to_csv( "thesis_data_2013.csv", index=False, encoding='utf-8-sig')


# In[ ]:





# In[5]:


#look at collumns of full NaN values and drop those

#the full set is 522 columns
data2012 = pd.read_csv('thesis_data_2012.csv')
print (data2012.shape)

#only 148 columns are not full NaN values
data2012_noNaN = data2012.dropna(axis=1, how='all')
print (data2012_noNaN.shape)

#only 8 colums dont have NaN values, droppping all those collums would be too drastic, analysis needed
data2012_anyNaN = data2012.dropna(axis=1, how='any')
print (data2012_anyNaN.shape)


# In[6]:


#look at collumns of full NaN values and drop those

#the full set is 571 columns
data2013 = pd.read_csv('thesis_data_2013.csv')
print (data2013.shape)

#only 158 columns are not full NaN values
data2013_noNaN = data2013.dropna(axis=1, how='all')
print (data2013_noNaN.shape)

#only 8 colums dont have NaN values, droppping all those collums would be too drastic, analysis needed
data2013_anyNaN = data2013.dropna(axis=1, how='any')
print (data2013_anyNaN.shape)


# In[ ]:





# In[8]:


# Differences of collumns between years
y1 = []

for col in data2013.columns.values: 
    if (pd.Series(data2012.columns.values).isin([col])).any():
        y1.append(col)
        
#print(y)

print (len(y1))


workingData2012 = data2012[data2012.columns.intersection(y1)]
workingData2013 = data2013[data2013.columns.intersection(y1)]


#dataframe that should be used when using all of the data but with the same collumns, has NaN values
# 484 collumns would be comparable out of 571


# In[11]:


#Check for differences in columns where full NaNs have been deleted
y2 = []

for col in data2013_noNaN.columns.values:
    if (pd.Series(data2012_noNaN.columns.values).isin([col])).any():
        y2.append(col)
        
print(len(y2))        
print(y2)

workingData_noNaN2012 = data2012_noNaN[data2012_noNaN.columns.intersection(y2)]
get_ipython().run_line_magic('store', 'workingData_noNaN2012')

workingData_noNaN2013 = data2013_noNaN[data2013_noNaN.columns.intersection(y2)]
get_ipython().run_line_magic('store', 'workingData_noNaN2013')

workingData_noNaN2012.to_csv( "workingData_noNaN2012.csv", index=False, encoding='utf-8-sig')
workingData_noNaN2013.to_csv( "workingData_noNaN2013.csv", index=False, encoding='utf-8-sig')

#dataframe that should be used when comparing noNaN years,most likekly canidates for ML training
#there would be 129 comparable collumns, out of 158 


# In[10]:


#Check for differences in columns where any NaNs have been deleted
y3 = []

for col in data2013_anyNaN.columns.values:
    if (pd.Series(data2012_anyNaN.columns.values).isin([col])).any():
        y3.append(col)
        
print(len(y3))
print(y3)

#there would be 8 comparable collumns out of 8


workingData_anyNaN2012 = data2012_anyNaN[data2012_anyNaN.columns.intersection(y3)]
workingData_anyNaN2013 = data2013_anyNaN[data2013_anyNaN.columns.intersection(y3)]

print (workingData_anyNaN2013)


#dataframe that should be used when comparing anyNaN years


# In[ ]:


# should huishodenR also interesect? as 25% of the population is replaced between 2012 and 2012


# In[ ]:




