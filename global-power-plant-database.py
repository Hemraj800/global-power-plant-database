#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing python libraries      

import pandas as pd                  # for data manipulation

import numpy as np                   # for mathematical calculations

import seaborn as sns                # for data visualization

import matplotlib.pyplot as plt      # for plotting graphs
plt.style.use('seaborn')             # the seaborn stylesheet will make our plots look neat and pretty.

get_ipython().run_line_magic('matplotlib', 'inline')
# "%matplotlib inline" ensures commands in cells below the cell that outputs a plot does not affect the plot
    
import warnings                      # to ignore any warnings
warnings.filterwarnings("ignore")


# In[2]:


database = pd.read_csv(r"C:\Users\HP\Downloads\database_IND.csv")


# In[3]:


database


# In[4]:


database.head()


# In[5]:


database.columns


# In[6]:


sns.pairplot(database)


# In[7]:


database[database.columns[:11]].describe()


# In[8]:


database.isna().any()


# In[9]:


database.head()


# In[10]:


sns.heatmap(database.isnull())


# In[11]:


database.dtypes


# In[13]:


database.drop(['country'],axis=1,inplace=True)
database.head()


# In[14]:


database.columns


# In[15]:


database['country_long'].unique()


# In[16]:


database['name'].unique()


# In[17]:


database['gppd_idnr'].unique()


# In[18]:


database['capacity_mw'].unique()


# In[20]:


sns.histplot(database['generation_gwh_2016'])
plt.show()


# In[21]:


sns.histplot(database['primary_fuel'])
plt.show()


# In[22]:


sns.histplot(database['generation_data_source'])
plt.show()


# In[23]:


from sklearn.preprocessing import LabelEncoder # import


# In[24]:


database.dtypes


# In[26]:


le=LabelEncoder()
for i in database.drop(['country_long'],axis=1):
    database[i]=le.fit_transform(database[i])
database


# In[27]:


database.head()


# In[28]:


database.dtypes


# In[31]:


# plot graph for co-relation in Bi Variate Analysis
import seaborn as sns
for col in database.drop(['generation_gwh_2015'],axis=1):
    plt.figure(figsize=(6,4))
    plt.title(f'{col} vs. generation_gwh_2015')
    sns.scatterplot(y=database[col],x=database['generation_gwh_2015'],hue=database['generation_gwh_2015'])
    plt.show()


# In[33]:


plt.figure(figsize=(6,4))
sns.catplot(x='latitude',y='longitude',data=database,kind='bar')
plt.show()


# In[34]:


database.corr()


# In[35]:


plt.figure(figsize=(15,8))
sns.heatmap(database.corr(),annot=True)


# In[ ]:




