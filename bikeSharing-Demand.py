import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
import seaborn as sns



style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
df=train.copy()
test_df=test.copy()


# In[85]:


df.head()
df.info()
df.columns
df['weather'].unique()


# In[89]:


df['season'].unique()


# /* A SHORT DESCRIPTION OF THE FEATURES.
# datetime - hourly date + timestamp
# 
# season - 1 = spring, 2 = summer, 3 = fall, 4 = winter
# 
# holiday - whether the day is considered a holiday
# 
# workingday - whether the day is neither a weekend nor holiday
# 
# weather -
# 
# 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# 
# 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# 
# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# 
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# 
# temp - temperature in Celsius
# 
# atemp - "feels like" temperature in Celsius
# 
# humidity - relative humidity
# 
# windspeed - wind speed
# 
# casual - number of non-registered user rentals initiated
# 
# registered - number of registered user rentals initiated */

# In[90]:


df.isnull().sum()
import missingno as msno
msno.matrix(df)


# In[93]:


df.season.value_counts()


# In[94]:


df.weather.value_counts()


# In[95]:


sns.factorplot(x='season',data=df,kind='count')


# In[96]:


sns.factorplot(x='weather',data=df,kind='count')


# In[97]:


sns.factorplot(x='workingday',data=df,kind='count')


# In[98]:


df.describe()


# In[99]:


sns.boxplot(data=df[['temp',
       'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']])
fig=plt.gcf()
fig.set_size_inches(10,10)


# In[100]:


df.temp.unique()


# In[101]:


cor_mat=df[:].corr()


# In[102]:


mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False


# In[103]:


fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)


# In[104]:


weather=pd.get_dummies(df['weather'],prefix='weather')
df=pd.concat([df,weather],axis=1)
weather=pd.get_dummies(test_df['weather'],prefix='weather')
test_df=pd.concat([test_df,weather],axis=1)
df.head()


# In[105]:


test_df.head()


# In[106]:


season=pd.get_dummies(df['season'],prefix='season')
df=pd.concat([df,season],axis=1)
season=pd.get_dummies(test_df['season'],prefix='season')
test_df=pd.concat([test_df,season],axis=1)


# In[107]:


df.head()


# In[108]:


df.drop(['season','weather'],inplace=True,axis=1)
test_df.drop(['season','weather'],inplace=True,axis=1)
df.head()


# In[109]:


df["hour"] = [t.hour for t in pd.DatetimeIndex(df.datetime)]
df["day"] = [t.dayofweek for t in pd.DatetimeIndex(df.datetime)]
df["month"] = [t.month for t in pd.DatetimeIndex(df.datetime)]
df['year'] = [t.year for t in pd.DatetimeIndex(df.datetime)]
df['year'] = df['year'].map({2011:0, 2012:1})
df.head()


# In[110]:


test_df["hour"] = [t.hour for t in pd.DatetimeIndex(test_df.datetime)]
test_df["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_df.datetime)]
test_df["month"] = [t.month for t in pd.DatetimeIndex(test_df.datetime)]
test_df['year'] = [t.year for t in pd.DatetimeIndex(test_df.datetime)]
test_df['year'] = test_df['year'].map({2011:0, 2012:1})
test_df.head()


# In[111]:


df.drop('datetime',axis=1,inplace=True)
df.head()


# In[144]:


corr_mat=df[:].corr()
mask=np.array(corr_mat)
mask[np.tril_indices_from(mask)]=False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=corr_mat,mask=mask,square=True,annot=True,cbar=True)


# In[145]:


df.drop(['casual','registered'],axis=1,inplace=True)
df.head()


# In[146]:


sns.factorplot(x='month',y='count',data=df,kind='bar')


# In[147]:


df_cpy=df.copy()
df_cpy.temp.describe()


# In[148]:


df_cpy.temp.unique()


# In[149]:


df_cpy['temp_bin']=np.floor(df_cpy['temp'])//5
df_cpy.temp_bin.unique()


# In[150]:


df_cpy.humidity.describe()


# In[151]:


df_cpy['hum_bins']=np.floor(df_cpy['humidity'])//10
df_cpy.hum_bins.unique()


# In[152]:


sns.factorplot(x='temp_bin',y='count',data=df_cpy,kind='bar',size=5,aspect=2)


# In[153]:


sns.factorplot(x='hum_bins',y='count',data=df_cpy,kind='bar',size=5,aspect=2)


# In[154]:


sns.scatterplot(x='temp',y='humidity',data=df_cpy)


# In[155]:


df.columns.to_series().groupby(df.dtypes).groups


# In[156]:


from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


# In[157]:


from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import mean_squared_log_error,mean_squared_error,r2_score,mean_absolute_error


# In[158]:


df.head()


# In[159]:


x_train,x_test,y_train,y_test=train_test_split(df.drop('count',axis=1),df['count'],test_size=0.25,random_state=42)


# In[160]:


models=[RandomForestRegressor(),AdaBoostRegressor(),BaggingRegressor(),KNeighborsRegressor(),SVR()]
model_names=['RandomForestRegressor','AdaBoostRegressor','BaggingRegressor','KNeighborsRegressor','SVR']
rmsle=[]
d={}
for model in range (len(models)):
    c=models[model]
    c.fit(x_train,y_train)
    test_predict=c.predict(x_test)
    rmsle.append(np.sqrt(mean_squared_log_error(test_predict,y_test)))
d={'modelling_algo':model_names,'RMSLE':rmsle}
d


# In[161]:


rmsle_frame=pd.DataFrame(d)
rmsle_frame


# In[162]:


sns.factorplot(y='modelling_algo',x='RMSLE',data=rmsle_frame,kind='bar',size=6,aspect=2)


# In[163]:


sns.factorplot(x='modelling_algo',y='RMSLE',data=rmsle_frame,kind='point',size=6,aspect=2)


# In[164]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# In[165]:


no_of_test=[500]
params_dict={'n_estimators':no_of_test,'n_jobs':[-1],'max_features':['auto','sqrt','log2']}
m=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params_dict,scoring='neg_mean_squared_error')
m.fit(x_train,y_train)
pred=m.predict(x_test)
print(np.sqrt(mean_squared_log_error(pred,y_test)))


# In[166]:


m.best_params_


# In[167]:


n_neighbors=[]
for i in range(0,50,5):
    if(i!=0):
        n_neighbors.append(i)
params_dict={'n_neighbors':n_neighbors,'n_jobs':[-1]}
m_knn=GridSearchCV(estimator=KNeighborsRegressor(),param_grid=params_dict,scoring='neg_mean_squared_log_error')
m_knn.fit(x_train,y_train)
pred_knn=m_knn.predict(x_test)
print(np.sqrt(mean_squared_log_error(pred_knn,y_test)))


# In[168]:


m_knn.best_params_


# In[169]:


test_df.head()


# In[170]:


x_train.head()


# In[171]:


pred=m.predict(test_df.drop('datetime',axis=1))
d={'datetime':test_df['datetime'],'count':pred}
ans=pd.DataFrame(d)
ans.head()


# In[172]:


ans.to_csv('answer_bike_share_demand.csv',index=False)


# THE END
