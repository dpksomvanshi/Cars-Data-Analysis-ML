#!/usr/bin/env python
# coding: utf-8

# In[1]:


from warnings import filterwarnings
filterwarnings('ignore')


# In[2]:


import os
os.chdir('D:/Datasets')


# In[3]:


import pandas as pd
df=pd.read_csv('Cars93.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.columns


# ### seperate dependant and independant features
# Weight ~ Remaining Features

# In[7]:


X=df.drop(labels=['id','Weight'], axis=1)
Y=df[['Weight']]


# In[8]:


X.shape


# In[9]:


X.head()


# In[10]:


Y.head()


# ## Build SKlearn pipeline for preprocessing

# In[11]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


# In[12]:


#sepearte cat,con features
from PM8 import catconsep
cat,con = catconsep(X)


# In[13]:


cat


# In[14]:


con


# In[15]:


#Build Numeric Pipleine
num_pipe = Pipeline(steps=[('Imputer',SimpleImputer(strategy='mean')),
                          ('Standard',StandardScaler())])
#categorical
cat_pipe= Pipeline(steps=[('Imputer',SimpleImputer(strategy='most_frequent')),
                          ('OHE',OneHotEncoder(handle_unknown='ignore'))])
#combine
pre = ColumnTransformer([('num',num_pipe,con),
                        ('cat',cat_pipe,cat)])


# In[16]:


X_pre = pre.fit_transform(X).toarray()
X_pre


# In[17]:


cols = pre.get_feature_names_out()
cols


# In[18]:


#create a Dataframe
X_pre = pd.DataFrame(X_pre,columns=cols)
X_pre.head()


# In[19]:


#train-test split
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X_pre,Y,test_size=0.2,random_state=75)


# In[20]:


#build base model - linear regreassion
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain,ytrain)


# In[21]:


#r2 score in training
r2_score_tr = model.score(xtrain,ytrain)
r2_score_tr


# In[22]:


#r2 score in testing
r2_score_ts =model.score(xtest,ytest)
r2_score_ts


# ## model evaluation with train-test split

# In[23]:


from PM8 import evaluate_model
evaluate_model(xtrain,ytrain,xtest,ytest,model)


# ## cross validation

# In[24]:


from sklearn.model_selection import cross_val_score


# In[25]:


scores = cross_val_score(model,xtrain,ytrain,cv=5,scoring='neg_mean_squared_error')
scores


# In[26]:


scores.mean()           #mean squared error


# In[27]:


mae_score = cross_val_score(model,xtrain,ytrain,cv=5,scoring='neg_mean_absolute_error')
mae_score


# In[28]:


mae_score.mean()


# In[29]:


r2_scores = cross_val_score(model,xtrain,ytrain,cv=5,scoring='r2')
r2_scores


# In[30]:


r2_scores.mean()


# In[31]:


neg_rmse = cross_val_score(model,xtrain,ytrain,cv=5,scoring='neg_root_mean_squared_error')
neg_rmse


# In[32]:


neg_rmse.mean()


# In[33]:


## 10 Fold CV
r2_s2= cross_val_score(model,xtrain,ytrain,cv=10,scoring='r2')
r2_s2


# In[34]:


r2_s2.mean()


# ## Ridge and Lasso Tuning
# 1. Alpha value : Hyperparameter
# 2. GridSearchCV

# In[35]:


# alpha values
import numpy as np
alphas=np.arange(0.1,60,0.1)
alphas


# # parameter
# ~~~
# model = Ridge(alpha=value)
# ~~~

# In[36]:


params = {'alpha':alphas}              #dictionary


# In[37]:


## GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge                     ##Ridge Model Evaluation
model1= Ridge()
gscv = GridSearchCV(model1,param_grid=params,cv=5,scoring='neg_root_mean_squared_error')
gscv.fit(xtrain,ytrain)


# In[38]:


gscv.best_params_


# In[39]:


gscv.best_score_


# In[40]:


best_ridge= gscv.best_estimator_
best_ridge


# In[41]:


# check model score in training
best_ridge.score(xtrain,ytrain)


# In[42]:


#Check score in testing
best_ridge.score(xtest,ytest)


# In[43]:


evaluate_model(xtrain,ytrain,xtest,ytest,best_ridge)


# In[44]:


## Tune Lasso Model
from sklearn.linear_model import Lasso
model2 = Lasso()
gscv2 = GridSearchCV(model2,param_grid=params,cv=5,scoring='neg_mean_squared_error')
gscv2.fit(xtrain,ytrain)


# In[45]:


gscv2.best_params_


# In[46]:


gscv2.best_score_


# In[47]:


best_lasso = gscv2.best_estimator_
best_lasso


# In[48]:


best_lasso.score(xtrain,ytrain)


# In[49]:


best_lasso.score(xtest,ytest)


# In[50]:


evaluate_model(xtrain,ytrain,xtest,ytest,best_lasso)


# In[51]:


# here I am selecting Ridge as my final model as testing score is higher


# In[52]:


# Estimate car weights from new csv file
df2=pd.read_csv('sample.csv')
df2


# In[53]:


xnew = pre.transform(df2).toarray()
xnew


# In[54]:


xnew = pd.DataFrame(xnew,columns=cols)
xnew


# In[55]:


## predict final weights for new cars


# In[56]:


preds= best_ridge.predict(xnew)
preds


# In[58]:


df3 = df2[['Manufacturer','Model']]
df3


# In[59]:


df3['pred_Weights'] = preds
df3


# In[60]:


df3.to_csv('Cars_predicted.csv',index=False)    #save file in csv


# In[ ]:




