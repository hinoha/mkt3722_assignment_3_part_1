#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import statistics
import statsmodels.formula.api as smf
import numpy.random as rd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_spss("InternetUsage_Assignment3.sav")
print(data)


# In[35]:


data[data.isnull().any(axis=1)]
print(data.isnull().sum())


# In[36]:


#Question 1: Are the respondents familiar with Internet?

familiar_data_cleanup = {"familiar": {"Very Familiar": int(6), "Somewhat unfamiliar": int(2)}}

data.replace(familiar_data_cleanup, inplace=True)

data.head(60)


# In[33]:


data.sort_values(by=["familiar"], inplace=True)

print(data)


# In[7]:


data["familiar"].value_counts()


# In[8]:


data.sort_values(by=["iattitude"], inplace=True)

print(data)


# In[9]:


#Question 2: Do they have a favorable attitude towards Internet? 

iattitude_data_cleanup = {"iattitude": {"Very Favorable": int(5)}}

data.replace(iattitude_data_cleanup, inplace=True)

data["iattitude"]


# In[10]:


data["iattitude"].value_counts()


# In[11]:


#Question 3: Do the respondents have a more favorable attitude towards Internet than towards technology? 

tattitude_data_cleanup = {"tattitude": {"Very Favorable": int(5), "somewhat unfamiliar": int(2)}}

data.replace(tattitude_data_cleanup, inplace=True)

data["tattitude"]


# In[12]:


data["tattitude"].value_counts()


# In[13]:


data.head()


# In[37]:


#4. Do male respondents have same familiarity towards Internet as female respondents?
data.sort_values(by=["sex"], inplace=True)

print(data)


# In[15]:


data[["familiar", "sex"]]


# In[16]:


data.groupby(["familiar", "sex"]).size()


# In[39]:


#5.	Group respondents into three user types: 
# a.	lighter users (weekly internet usage < 5 hours)
# b.	medium users (weekly internet usage >=5 but <9)
# c.	heavy users (weekly internet usage >=9 hours)
#What are the percentages of each type users? Do different types of users have same attitude towards Internet?


data[["iusage"]]


# In[18]:


data.groupby(["iusage"]).size()


# In[19]:


iusage_group = {"iusage": {int(2):"lighter users", int(3):"lighter users", int(4):"lighter users", int(5):"medium users", int(6):"medium users", int(7):"medium users", int(8):"medium users", int(9):"heavy users", int(10):"heavy users", int(11):"heavy users", int(12):"heavy users", int(13):"heavy users", int(14):"heavy users", int(15):"heavy users", int(16):"heavy users" }}

data.replace(iusage_group, inplace=True)

data.head(60)


# In[20]:


data.groupby(["iusage"]).size()


# In[40]:


#6. Is Internet shopping correlated with gender? 
data[["ishopping", "sex"]]


# In[22]:


data.head()


# In[23]:


data.corr("pearson")


# In[24]:


ishopping_dummy = pd.get_dummies(data, columns=["ishopping"])
pd.get_dummies(data, columns=["ishopping"]).head()


# In[25]:


sex_dummy = pd.get_dummies(data, columns=["sex"])
pd.get_dummies(data, columns=["sex"]).head()


# In[30]:


corr = np.array([ishopping_dummy,sex_dummy])


# In[27]:


from scipy import stats

stats.chi2_contingency(corr)


# In[28]:


chi2_stat, p_val, dof, ex = stats.chi2_contingency(corr)
print("===Chi2 Stat===")
print(chi2_stat)
print("\n")
print("===Degrees of Freedom===")
print(dof)
print("\n")
print("===P-Value===")
print(p_val)
print("\n")
print("===Contingency Table===")
print(ex)


# In[ ]:




