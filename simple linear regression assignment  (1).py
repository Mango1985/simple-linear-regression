#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


data=pd.read_csv("delivery_time.csv")
data.head()


# In[4]:


import warnings
warnings .filterwarnings('ignore')


# In[5]:


data.info()


# In[6]:


data.corr()


# In[7]:


import seaborn as sns


# In[8]:


sns.distplot(data['Delivery Time'])


# In[9]:


sns.distplot(data['Sorting Time'])


# In[10]:


import statsmodels.formula.api as smf


# In[11]:


Deliverytime= pd.Series(data.iloc[:,0])
Deliverytime


# In[12]:


Sortingtime= pd.Series(data.iloc[:,1])
Sortingtime


# In[13]:


model=smf.ols("Deliverytime~Sortingtime",data=data).fit()


# In[14]:


model.summary()


# In[15]:


model.params


# In[16]:


sns.regplot(x="Delivery Time", y="Sorting Time", data=data)


# In[17]:


model.resid_pearson


# In[18]:


# 2. Salary_hike
salary=pd.read_csv("Salary_Data.csv")
salary


# In[19]:


import warnings
warnings .filterwarnings('ignore')


# In[20]:


salary.describe()


# In[21]:


import seaborn as sns
sns.distplot(salary['YearsExperience'])


# In[22]:


sns.distplot(salary['Salary'])


# In[23]:


import statsmodels.formula.api as smf
sns.regplot(x="YearsExperience", y="Salary", data=salary)


# In[24]:


Yearsexperience= pd.Series(salary.iloc[:,0])
Yearsexperience


# In[25]:


Salary= pd.Series(salary.iloc[:,1])
Salary


# In[26]:


model=smf.ols("Yearsexperience~Salary",data=salary).fit()


# In[27]:


model.summary()


# In[29]:


import statsmodels.formula.api as smf
sns.regplot(x="YearsExperience", y="Salary", data=salary)


# In[30]:


print(model.tvalues, '\n', model.pvalues)

