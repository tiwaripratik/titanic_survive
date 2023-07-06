#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# In[2]:


titanic=pd.read_csv("D:\\ML\\dataset\\titanic\\train.csv")


# In[3]:


titanic.head()


# In[4]:


titanic.shape


# In[5]:


titanic.isnull().sum()


# In[6]:


titanic.info()


# In[7]:


titanic=titanic.drop('Cabin',axis=1)


# In[8]:


titanic.head()


# In[9]:


titanic['Age'].fillna(titanic['Age'].mean(),inplace=True)


# In[10]:


titanic.head()


# In[11]:


titanic.isnull().sum()


# In[12]:


print(titanic['Embarked'].mode())


# In[13]:


print(titanic['Embarked'].mode()[0])


# In[14]:


titanic['Embarked'].fillna(titanic['Embarked'].mode()[0],inplace=True)


# In[15]:


titanic


# In[16]:


titanic.isnull().sum()


# In[17]:


titanic.describe()


# In[18]:


titanic['Survived'].value_counts()


# In[19]:


sns.set()


# In[20]:


plt.figure(figsize=(7,7))
sns.countplot('Survived',data=titanic)


# In[21]:


titanic['Sex'].value_counts()


# In[22]:


plt.figure(figsize=(7,7))
sns.countplot('Sex',data=titanic)


# In[23]:


sns.countplot('Sex',hue='Survived',data=titanic)


# In[24]:


plt.figure(figsize=(7,7))
sns.countplot('Pclass',data=titanic)
plt.show()


# In[25]:


sns.countplot('Pclass',hue='Survived',data=titanic)


# In[26]:


sns.countplot('Embarked',hue='Survived',data=titanic)


# In[27]:


titanic.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)


# In[28]:


titanic.head()


# In[29]:


X=titanic.drop(['Name','Ticket','PassengerId','Survived'],axis=1)


# In[30]:


X


# In[31]:


Y=titanic['Survived']


# In[32]:


Y


# In[33]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)


# In[34]:


model=LogisticRegression()


# In[35]:


model.fit(X_train,Y_train)


# In[36]:


training_data=model.predict(X_train)


# In[37]:


training_accuracy=accuracy_score(training_data,Y_train)


# In[38]:


print("The accuracy of the training data is:",training_accuracy)


# In[39]:


testing_data=model.predict(X_test)


# In[40]:


testing_accuracy=accuracy_score(testing_data,Y_test)


# In[41]:


print("The accuracy of the testing data is:",testing_accuracy)


# In[ ]:




