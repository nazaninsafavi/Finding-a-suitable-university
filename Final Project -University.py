#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import classification_report
from sklearn import metrics


# In[3]:


df=pd.read_csv('university.csv')


# In[4]:


# preprocessings :


# In[5]:


for i in range(len(df['program'])):
    try:
        (df['program'][i]=='MS')
    except ValueError:
        df.loc[i, 'program'] = np.nan 


# In[6]:


for i in range(len(df['department'])):
    try:
        str(df['department'][i])
    except ValueError:
        df.loc[i, 'department'] = np.nan 


# In[7]:


for i in range(len(df['ugCollege'])):
    try:
        str(df['ugCollege'][i])
    except ValueError:
        df.loc[i, 'ugCollege'] = np.nan


# In[8]:


for i in range(len(df['univName'])):
    try:
        str(df['univName'][i])
    except ValueError:
        df.loc[i, 'univName'] = np.nan


# In[9]:


for i in range(len(df['greV'])):
    try:
        int(df['greV'][i]) 
    except ValueError:
        df.loc[i, 'greV'] = np.nan 


# In[10]:


for i in range(len(df['greV'])):
    if ( df['greV'][i] <130 or df['greV'][i]>170):
         df['greV'][i]= np.nan


# In[11]:


for i in range(len(df['greQ'])):
    try:
        int(df['greQ'][i]) 
    except ValueError:
        df.loc[i, 'greQ'] = np.nan


# In[12]:


for i in range(len(df['greQ'])):
    if ( df['greQ'][i] <130 or df['greQ'][i]>170):
         df['greQ'][i]= np.nan 


# In[13]:


for i in range(len(df['greA'])):
    try:
        int(df['greA'][i]) 
    except ValueError:
        df.loc[i, 'greA'] = np.nan


# In[14]:


for i in range(len(df['greA'])):
    if (  df['greA'][i]> 6):
         df['greA'][i]= np.nan 


# In[15]:


for i in range(len(df['toeflScore'])):
    try:
        int(df['toeflScore'][i]) 
    except ValueError:
        df.loc[i, 'toeflScore'] = np.nan


# In[17]:


for i in range(len(df['toeflScore'])):
    if ( df['toeflScore'][i] >120):
         df['toeflScore'][i]= np.nan 


# In[18]:


for i in range(len(df['cgpa'])):
    try:
        int(df['cgpa'][i]) 
    except ValueError:
        df.loc[i, 'cgpa'] = np.nan


# In[19]:


for i in range(len(df['cgpa'])):
    if (df['cgpa'][i] >100):
         df['cgpa'][i]= np.nan 


# In[20]:


for i in range(len(df['topperCgpa'])):
    try:
        int(df['topperCgpa'][i]) 
    except ValueError:
        df.loc[i, 'topperCgpa'] = np.nan


# In[21]:


for i in range(len(df['researchExp'])):
    try:
        int(df['researchExp'][i]) 
    except ValueError:
        df.loc[i, 'researchExp'] = np.nan


# In[22]:


for i in range(len(df['industryExp'])):
    try:
        int(df['industryExp'][i]) 
    except ValueError:
        df.loc[i, 'industryExp'] = np.nan


# In[23]:


for i in range(len(df['internExp'])):
    try:
        int(df['internExp'][i]) 
    except ValueError:
        df.loc[i, 'internExp'] = np.nan


# In[24]:


for i in range(len(df['confPubs'])):
    try:
        int(df['confPubs'][i])
    except ValueError:
        df.loc[i, 'confPubs'] = np.nan 


# In[25]:


for i in range(len(df['journalPubs'])):
    try:
        int(df['journalPubs'][i]) 
    except ValueError:
        df.loc[i, 'journalPubs'] = np.nan 


# In[26]:


new = df['termAndYear'].str.split('-', n = 1, expand = True)
df['term']= new[0]
df['year']=new[1]
df.drop(columns =["termAndYear"], inplace = True)


# In[27]:


df['toeflScore'] = df['toeflScore'].fillna(df['toeflScore'].mean())


# In[28]:


df['greV'] = df['greV'].fillna(df['greV'].mean())


# In[29]:


df['greQ'] = df['greQ'].fillna(df['greQ'].mean())


# In[30]:


df['greA'] = df['greA'].fillna(df['greA'].mean())


# In[31]:


df=df.dropna(thresh=int(df.shape[0]* .9) ,axis=1)


# In[32]:


df=df.dropna(thresh=int(df.shape[1]* .9) ,axis=0)


# In[33]:


df=df.dropna()


# In[34]:


df['cgpa']=df['cgpa'].dropna()


# In[35]:


df[['journalPubs', 'confPubs','year']] = df[['journalPubs','confPubs','year']].astype(int)


# In[36]:


#drop duplicated rows
df=df[df.duplicated()== False]


# In[37]:


#Putting all (int) data values in cgpa column into a same scale
df['cgpa'] = df['cgpa'].apply(lambda x: x*100)


# In[38]:


df.loc[:,'cgpa']=df.loc[:,'cgpa']/df.loc[:,'cgpaScale']


# In[39]:


#putting greA into right scale of 170
df['greA']=df['greA'].apply(lambda x: x*170)


# In[40]:


df.loc[:,'greA']=df.loc[:,'greA']/6


# In[41]:


df=df.dropna()


# In[42]:


df_train=df.copy()


# In[43]:


X_features=df_train.copy()


# In[44]:


#Based on the standard deviation of every single column showing bellow, lower values were omitted to reduce dimension. 
df_train.describe()


# In[45]:


pscaler=preprocessing.PowerTransformer()
df_train[['researchExp','industryExp','toeflScore','greA','cgpa','topperCgpa']] = pscaler.fit_transform(df_train[['researchExp',
            'industryExp','toeflScore','greA','cgpa','topperCgpa']])


# In[46]:


#removing outliers
rscaler = preprocessing.RobustScaler(with_scaling=True) 
df_train[['researchExp','industryExp','toeflScore','greA','cgpa','topperCgpa']] =rscaler.fit_transform(df_train[['researchExp',
            'industryExp','toeflScore','greA','cgpa','topperCgpa']] )


# In[47]:


scaler=preprocessing.StandardScaler()
df_train[['researchExp','industryExp','toeflScore','greA','cgpa','topperCgpa']] =scaler.fit_transform(df_train[['researchExp',
            'industryExp','toeflScore','greA','cgpa','topperCgpa']] )


# In[48]:


mscaler=preprocessing.MinMaxScaler()
df_train[['researchExp','industryExp','toeflScore','greA','cgpa','topperCgpa']] =mscaler.fit_transform(df_train[['researchExp',
            'industryExp','toeflScore','greA','cgpa','topperCgpa']] )


# In[49]:


# String columns one Hot encoding 
df_train=pd.get_dummies(df,columns=['major','program','department','ugCollege','univName','term'])


# In[50]:


X = df_train.drop(axis=1, columns=['userName','userProfileLink','greV','greQ','internExp','journalPubs','confPubs','admit'])
y = df_train['admit']


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.2)


# In[52]:


X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)


# In[53]:


from sklearn.ensemble import RandomForestClassifier


# In[54]:


RF=RandomForestClassifier(n_estimators=100)


# In[55]:


RF.fit(X_train,y_train)


# In[56]:


y_pred=RF.predict(X_test)


# In[57]:


print(metrics.accuracy_score(y_test,y_pred))


# In[58]:


print(classification_report(y_test, y_pred))


# In[59]:


y_pred=pd.DataFrame(RF.predict_proba(X_test), columns=['Rejection Probability','Admission Probability'])  


# In[60]:


y_pred.sort_values(by='Admission Probability',ascending=False,inplace=True)


# In[61]:


index_values=y_pred.index.values


# In[62]:


uniname=X_features['univName'].tolist()


# In[63]:


Series1=[]
for i in index_values:
    Series1.append(uniname[i])     


# In[64]:


Series2=pd.Series(Series1,index=index_values)


# In[65]:


dict1=Series2.to_dict()


# In[66]:


y_pred['university name']=dict1.values()


# In[68]:


y_pred.drop_duplicates(subset='university name',inplace=True)


# In[72]:


# The Final Output: Showing 5 universities with the highest Probability of Admission 

y_prediction=y_pred.head(5)
y_prediction


# In[ ]:




