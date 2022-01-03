#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import torch
import pandas as pd
import matplotlib
import tensorflow as tf
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import timedelta
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow import keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.layers import Bidirectional


# In[38]:


## read the data 
df = pd.read_csv("T1.csv")


# In[39]:


df['Date/Time'] = pd.to_datetime(df['Date/Time'], format ="%d %m %Y %H:%M")
df = df.rename(columns={'Date/Time': 'Date','LV ActivePower (kW)':'ActivePower','Wind Speed (m/s)':'WindSpeed',
                        'Theoretical_Power_Curve (KWh)':'TheoreticalPowerCurve','Wind Direction (°)':'WindDirection'})
df


# In[40]:


import time


# In[41]:


df.fillna(0)


# In[42]:


df.drop(df[(df['WindSpeed'].isin([0]))].index, inplace = True)
df


# In[43]:


df.drop(df[(df['WindSpeed']>20)].index, inplace = True)
df


# In[44]:


df.drop(df[(df['Date'].isin([0]))].index, inplace = True)
df


# In[45]:


df.drop(df[(df['ActivePower']==0) & (df['TheoreticalPowerCurve']==0)].index, inplace = True)


# In[73]:


df.drop(df[(df['ActivePower']<0)].index, inplace = True)


# In[74]:


df=df.reset_index(drop=True)
df


# In[75]:


my_rho1 = np.corrcoef(df['ActivePower'], df['WindSpeed'])
my_rho1


# In[76]:


my_rho2 = np.corrcoef(df['ActivePower'], df['TheoreticalPowerCurve'])
my_rho2


# In[77]:


my_rho3 = np.corrcoef(df['ActivePower'], df['WindDirection'])
my_rho3


# In[82]:


plt.scatter(df['WindSpeed'], df['ActivePower'],s=3,alpha=0.3)
plt.xlabel('WindSpeed'); plt.ylabel('ActivePower'); 
plt.legend


# In[83]:


plt.scatter(df['TheoreticalPowerCurve'], df['ActivePower'],s=3,alpha=0.3)
plt.xlabel('TheoreticalPowerCurve'); plt.ylabel('ActivePower'); 
plt.legend


# In[84]:


def swinging_door(data_x,data_y,data_z, E):
        first_i = 0  #index of door
        i_U = 1 #index of last element of upper line
        i_L = 1 #index of last element of lower line
        i = 2 #index of new point
        results_x = []
        results_y = []
        results_z = []
        U = data_y[first_i] * (1+E)
        L = data_y[first_i] * (1-E)

        while i < len(data_x):

                k_U = ((U - data_y[i_U]) / (data_x[first_i] - data_x[i_U]))

                k_L = ((L - data_y[i_L]) / (data_x[first_i] - data_x[i_L]))

                LINE_U_Y = k_U*data_x[i] + (data_y[i_U] - k_U*data_x[i_U]) #highst of upper line near new point

                LINE_L_Y = k_L*data_x[i] + (data_y[i_L] - k_L*data_x[i_L]) #highst of lower line near new point

                if data_y[i] >= LINE_U_Y:
                        i_U = i # raised the line U higher

                if data_y[i] <= LINE_L_Y:
                        i_L = i # raised the line L higher

                if k_U < k_L: # compare more doors or not if more then opened
                        i = i + 1 
                else:  # if opened, we add 2 points the first, and the penultimate before the opening
                        results_x.append(data_x[first_i])
                        results_y.append(data_y[first_i])
                        results_z.append(data_z[first_i])
                        first_i = i - 1
                        i_U = i
                        i_L = i
                        i = i + 1
                        U = data_y[first_i] * (1+E) 
                        L = data_y[first_i] * (1-E)

        if len(results_x) == 0:
                results_x.append(data_x[first_i])
                results_y.append(data_y[first_i])
                results_z.append(data_z[first_i])
                results_x.append(data_x[i - 1])
                results_y.append(data_y[i - 1])
                results_z.append(data_z[i - 1])

    
        return results_x,results_y,results_z


# In[85]:


ramp = swinging_door(df.index,df['ActivePower'],df['Date'],0.3)


# In[86]:


df['r']=0


# In[87]:


for i in ramp[0]:
       for j in list(df.index):
            
            if i==j:
                df.loc[j,"r"]=1
                break


# In[88]:


rows = []

for _, row in tqdm(df.iterrows(),total=df.shape[0]):
    row_data = dict(
        day_of_week=row.Date.dayofweek,
        day_of_month=row.Date.day,
        week_of_year=row.Date.week,
        month=row.Date.month,
        
        ActivePower=row.ActivePower,
        WindSpeed=row.WindSpeed,
        TheoreticalPowerCurve=row.TheoreticalPowerCurve,
        RampEvent=row.r
        )
    rows.append(row_data)
    
features_df = pd.DataFrame(rows)


# In[89]:


features_df.shape


# In[90]:


features_df


# In[131]:


train_size = int(len(df)* .9)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

print(train.shape, test.shape)


# In[280]:


test


# ### LTSM

# In[92]:


pd.options.mode.chained_assignment = None


# In[93]:


f_columns = ['WindSpeed','TheoreticalPowerCurve']

f_transformer = RobustScaler()
ap_transformer = RobustScaler()

f_transformer = f_transformer.fit(train[f_columns].to_numpy())
ap_transformer = ap_transformer.fit(train[['ActivePower']])

train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
train['ActivePower'] = ap_transformer.transform(train[['ActivePower']])

test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
test['ActivePower'] = ap_transformer.transform(test[['ActivePower']])


# In[94]:


def create_dataset(X, y, time_steps=1):
    
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i: (i + time_steps)].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


# In[303]:


TIME_STEPS = 24

X_train, y_train = create_dataset(train,train.ActivePower, time_steps=TIME_STEPS)
X_test, y_test = create_dataset(test,test.ActivePower, time_steps=TIME_STEPS)


# In[96]:


print(X_train.shape, y_train.shape)


# In[97]:


print(X_test.shape, y_test.shape)


# In[98]:


model = Sequential()

model.add(
    Bidirectional(
        LSTM(
            units=128,
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )
)

model.add(Dropout(rate=0.2))
model.add(Dense(units=1))


# In[99]:


model.compile(optimizer='adam', loss='mse')


# In[100]:


X_train=np.asarray(X_train).astype(np.float32)
y_train=np.asarray(y_train).astype(np.float32)


# In[101]:


history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, shuffle=False)


# In[102]:


y_pred = model.predict(X_test)


# In[103]:


y_train_inv = ap_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = ap_transformer.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = ap_transformer.inverse_transform(y_pred)


# In[104]:


y_pred_inv.shape


# In[105]:


new_test = df.iloc[train_size:len(df)]
train_datas = new_test.iloc[0:len(y_pred_inv)]
train_dates = train_datas['Date']
train_dates


# In[106]:


y_preddata = np.squeeze(y_pred_inv)


# In[107]:


df_forecast = pd.DataFrame({'Date':train_dates, 'ActivePower':y_preddata})
df_forecast=df_forecast.reset_index(drop=True)
df_forecast


# In[108]:


ramp2 = swinging_door(df_forecast.index,df_forecast['ActivePower'],df_forecast['Date'],0.3)


# In[109]:


df_forecast['Ramp'] = 0


# In[110]:


for i in ramp2[0]:
       for j in list(df_forecast.index):
            
            if i==j:
                df_forecast.loc[j,"Ramp"]=1
                break


# In[111]:


test_power = new_test.iloc[0:len(y_pred_inv)]
test_power.reset_index(drop=True)


# In[112]:


# original = test_power[['Date', 'ActivePower','r']]
# original = original.loc[original['Date'] >= '2018-11-20 00:00:00']


# In[113]:


# original.drop(original[(original['r'].isin([0]))].index, inplace=True)


# In[326]:


plt.figure(figsize=(10,5))
plt.plot(test_power['Date'], test_power['ActivePower'], color='blue', label='True')
plt.plot(df_forecast['Date'], df_forecast['ActivePower'], color='red', linestyle = "--", label='LSTM')

plt.xticks(rotation = '45')
plt.xlabel('Time Series'); plt.ylabel('ActivePower'); plt.title('Actual and Predicted Values');
plt.legend()
plt.show


# In[114]:


plt.figure(figsize=(20,20))

ax1 = plt.subplot(311)
plt.plot(test_power['Date'], test_power['ActivePower'], color='grey', label='True')
plt.plot(df_forecast['Date'], df_forecast['ActivePower'], color='red', linestyle = "--", label='LSTM')

ax2 = plt.subplot(312)
plt.scatter(test_power['Date'], test_power['r'], marker='o',s=4.0, color='grey',label='True')
plt.plot(test_power['Date'], test_power['r'], linestyle='solid', color='grey',alpha=0.3)

ax3 = plt.subplot(313)
plt.scatter(df_forecast['Date'], df_forecast['Ramp'], marker='o',s=4.0, color='grey',label='LSTM')
plt.plot(df_forecast['Date'], df_forecast['Ramp'], linestyle='solid', color='grey',alpha=0.3)

ax1.legend()         
ax2.legend()         
ax3.legend()
plt.show()


# ### RandomForest

# In[192]:


rows = []

for _, row in tqdm(df.iterrows(),total=df.shape[0]):
    row_data = dict(
        day_of_week=row.Date.dayofweek,
        day_of_month=row.Date.day,
        week_of_year=row.Date.week,
        month=row.Date.month,
        
        ActivePower=row.ActivePower,
        WindSpeed=row.WindSpeed,
        TheoreticalPowerCurve=row.TheoreticalPowerCurve,
        
        )
    rows.append(row_data)
    
features_dfn = pd.DataFrame(rows)


# In[193]:


original = pd.DataFrame(features_dfn)

original


# In[194]:


train_size1 = int(len(df)* .9)
test_size1 = len(df) - train_size1
train1, test1 = df.iloc[0:train_size1], df.iloc[train_size1:len(df)]

print(train1.shape, test1.shape)


# In[144]:


features2 = original.drop(original[(original['WindSpeed'].isin([0]))].index)


# In[119]:


# original2 = df.drop(df[(df['r'].isin([0]))].index)


# In[120]:


# train_size2 = int(len(original2)* .9)
# test_size2 = len(original2) - train_size2
# train2, test2 = original2.iloc[0:train_size2], original2.iloc[train_size2:len(original2)]

# print(train2.shape, test2.shape)


# In[42]:


# from sklearn.preprocessing import MinMaxScaler


# In[43]:


# scaler = MinMaxScaler(feature_range=(-1, 1))
# scaler = scaler.fit(features2)


# In[44]:


# features2 = pd.DataFrame(
#     scaler.transform(features2),
#     index=features2.index,
#     columns=features2.columns
# )


# In[145]:


# Labels are the values we want to predict
labels = np.array(features2['ActivePower'])
# Remove the labels from the features
# axis 1 refers to the columns
features2= features2.drop('ActivePower', axis = 1)
# Saving feature names for later use
feature2_list = list(features2.columns)
# Convert to numpy array
features2 = np.array(features2)


# In[146]:


# Split the data into training and testing sets
train_X, test_X, train_Y, test_Y = train_test_split(features2, labels, test_size = 0.1, random_state = 42)


# In[124]:


from sklearn.ensemble import RandomForestRegressor


# In[147]:


# create an instance of the RandomForestClassifier
mdl = RandomForestRegressor(n_estimators=10,random_state=42)


# In[293]:


# Fit hte model
mdl.fit(train_X,train_Y)


# In[149]:


yTestPred = mdl.predict(test_X) # evaluate performance on test data


# In[150]:


# test2


# In[52]:


# RFtest = original2[train_size2:]


# In[53]:


# dates = RFtest.Date.tolist()


# In[54]:


# dates


# In[151]:


# Dataframe with predictions and dates
predictions_data = pd.DataFrame(data = {'date': test['Date'], 'ActivePower':yTestPred})
predictions_data


# In[152]:


RFtest = predictions_data.reset_index(drop=True)
RFtest


# In[57]:


# RFtest['r'] = 0


# In[153]:


ramp3 = swinging_door(RFtest.index,RFtest['ActivePower'],RFtest['date'],0.3)
RFtest['Ramp'] = 0


# In[59]:


# predictions_data.drop(predictions_data[(predictions_data['predictionr'].isin([0]))].index, inplace=True)


# In[154]:


for i in ramp3[0]:
       for j in list(RFtest.index):
            
            if i==j:
                RFtest.loc[j,"Ramp"]=1
                break


# In[155]:


RFtest


# In[156]:


testdata = test.reset_index(drop=True)
testdata


# In[157]:


plt.figure(figsize=(20,20))

ax1 = plt.subplot(311)
plt.plot(testdata['Date'], testdata['ActivePower'], color='grey', label='True')
plt.plot(RFtest['date'], RFtest['ActivePower'], color='blue', linestyle = "--", label='RandomForest')

ax2 = plt.subplot(312)
plt.scatter(testdata['Date'], testdata['r'], marker='o',s=4.0, color='grey',label='True')
plt.plot(testdata['Date'], testdata['r'], linestyle='solid', color='grey',alpha=0.3)

ax3 = plt.subplot(313)
plt.scatter(RFtest['date'], RFtest['Ramp'], marker='o',s=4.0, color='grey',label='RandomForest')
plt.plot(RFtest['date'], RFtest['Ramp'], linestyle='solid', color='grey',alpha=0.3)

ax1.legend()         
ax2.legend()         
ax3.legend()
plt.show()


# ### SVM

# In[195]:


#SVM
from sklearn.model_selection import RepeatedKFold


# In[196]:


# Repeat 5-fold cross-validation, ten times
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1040)
# set hyperprameter search grid
paramGrid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


# In[197]:


from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC


# In[198]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


X_train, y_train = create_dataset(train,train.ActivePower, time_steps=TIME_STEPS)
X_test, y_test = create_dataset(test,test.ActivePower, time_steps=TIME_STEPS)


# In[304]:


scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(X_train)


# In[307]:


len(X_train)


# In[305]:


train_svmx = scaler.transform(X_train)


# In[297]:


scaler = scaler.fit(X_test)


# In[298]:


test_svmx = scaler.transform(X_test)


# In[306]:



# SVM regularisation parameter
# svr = svm.SVR(kernel='linear').fit(train_svmx, train_Y)
# svr
rbf_svr = svm.SVR(kernel='rbf',C=100, gamma=0.1, epsilon=0.1).fit(X_train, y_train)
rbf_svr
# poly_svr = svm.SVR(kernel='poly', degree=3, C=1e4).fit(train_svmx, train_Y)
# poly_svr
# Prediction using the polynomial kernel
svrTestPred = rbf_svr.predict(X_test) # evaluate performance on test data


# In[203]:


svrTestPred


# In[204]:


# svcTestPred = svcmodel.predict(test_X) # evaluate performance on test data


# In[205]:


# Dataframe with predictions and dates
svcTestPred_data = pd.DataFrame(data = {'date': test['Date'],  'ActivePower':svrTestPred})


# In[206]:


SVRtest = svcTestPred_data.reset_index(drop=True)
SVRtest


# In[207]:


ramp4 = swinging_door(SVRtest.index,SVRtest['ActivePower'],SVRtest['date'],0.3)


# In[208]:


SVRtest['Ramp'] = 0


# In[209]:


for i in ramp4[0]:
       for j in list(SVRtest.index):
            
            if i==j:
                SVRtest.loc[j,"Ramp"]=1
                break


# In[210]:


plt.figure(figsize=(20,20))

ax1 = plt.subplot(311)
plt.plot(testdata['Date'], testdata['ActivePower'], color='grey', label='True')
plt.plot(SVRtest['date'], SVRtest['ActivePower'], color='green', linestyle = "--", label='SVM')

ax2 = plt.subplot(312)
plt.scatter(testdata['Date'], testdata['r'], marker='o',s=4.0, color='grey',label='True')
plt.plot(testdata['Date'], testdata['r'], linestyle='solid', color='grey',alpha=0.3)

ax3 = plt.subplot(313)
plt.scatter(SVRtest['date'], SVRtest['Ramp'], marker='o',s=4.0, color='grey',label='SVM')
plt.plot(SVRtest['date'], SVRtest['Ramp'], linestyle='solid', color='grey',alpha=0.3)

ax1.legend()         
ax2.legend()         
ax3.legend()
plt.show()


# In[ ]:





# ### ANN

# In[211]:


from torch import nn, optim
import torch.nn.functional as F


# In[219]:


rows = []

for _, row in tqdm(df.iterrows(),total=df.shape[0]):
    row_data = dict(
        day_of_week=row.Date.dayofweek,
        day_of_month=row.Date.day,
        week_of_year=row.Date.week,
        month=row.Date.month,
        
        ActivePower=row.ActivePower,
        WindSpeed=row.WindSpeed,
        TheoreticalPowerCurve=row.TheoreticalPowerCurve,
        )
    rows.append(row_data)
    
df2 = pd.DataFrame(rows)


# In[220]:


df2


# In[221]:


features3 = original.drop(original[(original['WindSpeed'].isin([0]))].index)


# In[222]:


features3


# In[189]:


cols_to_scale = ['Date','ActivePower','WindSpeed','TheoreticalPowerCurve']


# In[190]:


scaler = MinMaxScaler()
features3[cols_to_scale] = scaler.fit_transform(features3[cols_to_scale])


# In[223]:


features3.sample(3)


# In[86]:


features3.drop(['RampEvent'], axis=1, inplace=True)


# In[224]:


X = features3.drop('ActivePower',axis = 'columns')
y = features3['ActivePower']


# In[225]:


# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.1, random_state = 5)


# In[226]:


X_train.shape


# In[227]:


from tensorflow import keras as keras


# In[229]:


model = keras.Sequential([
    keras.layers.Dense(5, input_shape=(6,),activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']
             )

model.fit(X_train, Y_train, epochs=20)


# In[230]:


model.evaluate(X_test, Y_test)


# In[231]:


yp = model.predict(X_test)
yp


# In[232]:


yp_inv = ap_transformer.inverse_transform(yp)


# In[233]:


yp_inv


# In[234]:


# anny_pred = []
# for element in yp:
#     if element > 0.5:
#         anny_pred.append(1)
#     else:
#         anny_pred.append(0)


# In[235]:


ANNyp = np.squeeze(yp_inv)


# In[241]:


# Dataframe with predictions and dates
yp_data = pd.DataFrame(data = {'Date': test['Date'],  'ActivePower':ANNyp})


# In[244]:


ANNtest = yp_data.reset_index(drop=True)
ANNtest


# In[238]:


# yp_data.drop(yp_data[(yp_data['prediction'].isin([0]))].index, inplace=True)


# In[245]:


ramp5 = swinging_door(ANNtest.index,ANNtest['ActivePower'],ANNtest['Date'],0.3)
ANNtest['Ramp'] = 0


# In[246]:


for i in ramp5[0]:
       for j in list(ANNtest.index):
            
            if i==j:
                ANNtest.loc[j,"Ramp"]=1
                break


# In[247]:


plt.figure(figsize=(20,20))

ax1 = plt.subplot(311)
plt.plot(testdata['Date'], testdata['ActivePower'], color='grey', label='True')
plt.plot(ANNtest['Date'], ANNtest['ActivePower'], color='m', linestyle = "--", label='ANN')

ax2 = plt.subplot(312)
plt.scatter(testdata['Date'], testdata['r'], marker='o',s=4.0, color='grey',label='True')
plt.plot(testdata['Date'], testdata['r'], linestyle='solid', color='grey',alpha=0.3)

ax3 = plt.subplot(313)
plt.scatter(ANNtest['Date'], ANNtest['Ramp'], marker='o',s=4.0, color='grey',label='ANN')
plt.plot(ANNtest['Date'], ANNtest['Ramp'], linestyle='solid', color='grey',alpha=0.3)

ax1.legend()         
ax2.legend()         
ax3.legend()
plt.show()


# ### Convex multi-task feature learning

# In[248]:


from numpy import linalg as LA


# In[261]:


df2


# In[262]:


features_df2 = df2.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()


# In[268]:


X_train = pd.DataFrame(features_df2)
X_train.drop(['ActivePower','week_of_year','day_of_week','day_of_month','month','WindSpeed'], axis=1, inplace=True)
Y_train = pd.DataFrame(features_df2['ActivePower'])
Tasks = pd.DataFrame(features_df2['week_of_year'])


# In[264]:


class MLT_Task():
    def __init__(self, X, Y, gamma=1, l_rate=1e-5, n_epoch=15, N = 5, penalization=False, verbose=0):
        self.N = N # dimension representation space
        self.d = X[0].shape[1] # number of features
        self.gamma = gamma # penalization term
        self.l_rate = l_rate # learning rate
        self.n_epoch = n_epoch # number of epochs
        self.X = X # list of matrix of shape (m[i],d)
        self.Y = Y # list of matrix of shape (m[i],1)
        self.T = len(X) # number of tasks
        self.m = [X[i].shape[0] for i in range(len(X))] # number of individuals for each task, list of len(T)
        self.X_conc = np.vstack(X) # aggregated data on all tasks
        self.M = np.vstack(X).shape[0] # total number of individuals
        self.U_0 = np.random.rand(self.d,self.N) # initial representation matrix of shape (d,N)
        self.A_0 = np.random.rand(self.T,self.N) # initial regression matrix of shape (N,T)
        self.penalization = penalization
        self.verbose = verbose
        
        self.index_task = [0]*(self.T+1)# index of each task in global vector X_low and X_conc
        for t in range(1,self.T):
            self.index_task[t] = self.index_task[t-1] + self.m[t-1]
        self.index_task[self.T] = self.M
        
        self.reverse_index_task = np.zeros(self.M, dtype=int)
        for t in range(self.T):
            self.reverse_index_task[self.index_task[t]:self.index_task[t+1]] = t
        
        if self.verbose==1:
            print('_'*100)
            print('number of tasks (T) : %i' %self.T)
            print('features space lower dimension (N) : %i' %self.N)
            print('original feature space dimension (d) : %i\n' %self.d)
        
    def Loss_function(self, A, U):
        X_low = np.dot(self.X_conc, U) # data in lower dimensional space
        # regression, list of len(T) of matrix of shape (m[i],1)
        Y_hat = [X_low[self.index_task[t]:self.index_task[t+1],:] @ A[t] for t in range(self.T)]
        
        # compute loss
        L = 0
        for t in range(self.T):
            for i in range(self.m[t]):
                L += (Y_hat[t][i]-self.Y[t][i])**2 # average error across tasks
                
        # penalization term
        if self.penalization:
            A2 = LA.norm(A, 2, axis=1)
            A2 = LA.norm(A2, 2) # /!\ norm change, should be one ! here trace norm
            pen = self.gamma*A2
            L += pen
        
        return L, Y_hat

    def grad(self, A, U, t, i):
        grad_A_t = -(2*(self.Y[t][i] - self.X[t][i] @ U @ A[t])*(U.T @ self.X[t][i].T))
        grad_U = -(2 * (self.Y[t][i] - self.X[t][i] @ U @ A[t]) * (self.X[t][i].reshape(1, self.d).T @ A[t].reshape(self.N,1).T))
        grads = {"A_t": grad_A_t, "U": grad_U}
        return grads
    
    def explained_var(self, y_pred, y_train):
        #Computes the explained variance
        SS_err = np.linalg.norm(y_pred-y_train)**2/len(y_pred)
        SS_tot = np.var(y_train)
        return 1-SS_err/SS_tot

    # Estimate linear regression coefficients using stochastic gradient descent
    def opt(self):
        r = list(range(self.M))
        A = self.A_0
        U = self.U_0
        for epoch in range(self.n_epoch):
            random.shuffle(r)
            for i in r:
                t =  self.reverse_index_task[i]
                grad = self.grad(A, U, t, i-self.index_task[t])
                # coeff update
                A[t] = A[t] - self.l_rate * grad['A_t']
                U = U - self.l_rate * grad['U']
            self.l_rate = self.l_rate / (1 + epoch)
            loss, pred = self.Loss_function(A, U)
            Y_pred = np.hstack(pred)
            Y_train =  np.hstack(self.Y)
            Y_pred = Y_pred.reshape(Y_pred.shape[0],1)
            Y_train = Y_train.reshape(Y_train.shape[0],1)
            expl_var = self.explained_var(Y_pred, Y_train)
            if self.verbose==1:
                print('>epoch = %2d, loss = %3.2f, explained var = %3.2f' % (epoch, np.sqrt(loss/self.M), expl_var))
        if self.verbose==1:
            print('_'*100)
        return A, U, pred, np.sqrt(loss/self.M), expl_var
    
    def grad_F(self, A, U):
        grad_A_t = np.zeros(A.shape)
        grad_U = np.zeros(U.shape)
        for t in range(self.T):
            for i in range(self.m[t]):
                grad_A_t[t] += -(2*(self.Y[t][i] - self.X[t][i] @ U @ A[t]) * (U.T @ self.X[t][i].T))
                grad_U += -(2 * (self.Y[t][i] - self.X[t][i] @ U @ A[t]) * (self.X[t][i].reshape(1, self.d).T @ A[t].reshape(self.N,1).T))
        grads = {"A": grad_A_t, "U": grad_U}
        return grads
    
    def operator_prox(self, A, eps):
        u, s, v = np.linalg.svd(A)
        s=[max(s_-0.1,0) for s_ in s]
        ss = np.zeros((len(u),len(v)))
        l = min(len(u), len(v))
        ss[:l,:l] = np.diag(s)
        return u @ ss @ v
        
    def algo_prox(self, nb_iter=200):
        eps = 1e-9
        U = self.U_0
        A = self.A_0

        for it in range(nb_iter):
            grad = self.grad_F(A, U)
            U = U - eps * grad['U']
            A = A - eps * grad['A']
            A = self.operator_prox(A, eps)
            
            loss, pred = self.Loss_function(A, U)
            Y_pred = np.hstack(pred)
            Y_train =  np.hstack(self.Y)
            Y_pred = Y_pred.reshape(Y_pred.shape[0],1)
            Y_train = Y_train.reshape(Y_train.shape[0],1)
            expl_var = self.explained_var(Y_pred, Y_train)
            
            if self.verbose==1:
                print('>iter = %2d, loss = %3.2f, explained var = %3.2f' % (it, np.sqrt(loss/self.M), expl_var))
        
        return A, U, np.sqrt(loss/self.M), pred, expl_var


# In[269]:


# test parameters
X_train_list = [X_train[Tasks.week_of_year == t].values for t in Tasks.week_of_year.unique()]
Y_train_list = [Y_train[Tasks.week_of_year == t].values[:,0] for t in Tasks.week_of_year.unique()]

MLT_Task_test = MLT_Task(X_train_list, Y_train_list, N=28, n_epoch=10, l_rate=1e-5, verbose=0, penalization=True)
A, U, val, pred, var = MLT_Task_test.algo_prox()
print(pred)


# In[110]:


def flatten(a):
    if not isinstance(a, (list,)):
        return [a]
    else:
        b = []
        for item in a:
            b += flatten(item)
    return b


# In[111]:


import operator
from functools import reduce


# In[112]:


predcml = []
for i in range(0,52):
    for m in pred[i]:
        predcml.append(m)


# In[113]:


predcml_data = pd.DataFrame(data = {'date': df['Date'], 'ActivePower':predcml})


# In[114]:


new_predcml = predcml_data.iloc[train_size:len(df)]


# In[115]:


new_predcml=new_predcml.reset_index(drop=True)
new_predcml


# In[116]:


ramp6 = swinging_door(new_predcml.index,new_predcml['ActivePower'],new_predcml['date'],0.3)
new_predcml['Ramp'] = 0


# In[117]:


for i in ramp6[0]:
       for j in list(new_predcml.index):
            
            if i==j:
                new_predcml.loc[j,"Ramp"]=1
                break


# In[118]:


plt.figure(figsize=(20,20))

ax1 = plt.subplot(311)
plt.plot(testdata['Date'], testdata['ActivePower'], color='grey', label='True')
plt.plot(new_predcml['date'], new_predcml['ActivePower'], color='yellow', linestyle = "--", label='CMTL')

ax2 = plt.subplot(312)
plt.scatter(testdata['Date'], testdata['r'], marker='o',s=4.0, color='grey',label='True')
plt.plot(testdata['Date'], testdata['r'], linestyle='solid', color='grey',alpha=0.3)

ax3 = plt.subplot(313)
plt.scatter(new_predcml['date'], new_predcml['Ramp'], marker='o',s=4.0, color='grey',label='CMTL')
plt.plot(new_predcml['date'], new_predcml['Ramp'], linestyle='solid', color='grey',alpha=0.3)

ax1.legend()         
ax2.legend()         
ax3.legend()
plt.show()


# In[119]:


df_forecast['Date']


# In[ ]:





# In[ ]:





# In[120]:


df_forecast2 = df_forecast.drop(df_forecast[(df_forecast['Ramp'].isin([0]))].index)
RFtest2 = RFtest.drop(RFtest[(RFtest['r'].isin([0]))].index)
svcTestPred_data2 = svcTestPred_data.drop(svcTestPred_data[(svcTestPred_data['Ramp'].isin([0]))].index)
yp_data2 = yp_data.drop(yp_data[(yp_data['Ramp'].isin([0]))].index)
new_predcml2 = new_predcml.drop(new_predcml[(new_predcml['Ramp'].isin([0]))].index)


# In[125]:


test_power['r'].replace(1, 12, inplace=True)
df_forecast2['Ramp'].replace(1, 10, inplace=True)
RFtest2['r'].replace(1, 8, inplace=True)
svcTestPred_data2['Ramp'].replace(1, 6, inplace=True)
yp_data2['Ramp'].replace(1, 4, inplace=True)
new_predcml2['Ramp'].replace(1, 2, inplace=True)


# In[132]:


plt.figure(figsize=(10,10))
plt.plot(test_power['Date'], test_power['r'], color='grey',marker='.',alpha=0.5, label='true')


plt.scatter(df_forecast2['Date'], df_forecast2['Ramp'], color='red', marker='.',alpha=0.7, label='LSTM')
plt.scatter(RFtest2['date'], RFtest2['r'], color='blue',marker='.',alpha=0.7,label = 'RandomForest')
plt.scatter(svcTestPred_data2['date'], svcTestPred_data2['Ramp'], color='green',marker='.',alpha=0.7,label = 'SVM')
plt.scatter(yp_data2['Date'], yp_data2['Ramp'], color='m',marker='.',alpha=0.7,label = 'ANN')
plt.scatter(new_predcml2['date'], new_predcml2['Ramp'], color='yellow',marker='.',alpha=0.7,label = 'CMTL')

# Graph labels
plt.xticks(rotation = '45')
plt.xlabel('Date'); plt.ylabel('RampEvents'); plt.title('Actual and Predicted Values');
plt.legend(bbox_to_anchor=(1,0.5), loc=3,borderaxespad=1)


# In[314]:


test_power2 = test_power.drop(test_power[(test_power['r'].isin([0]))].index)
df_forecast2 = df_forecast.drop(df_forecast[(df_forecast['Ramp'].isin([0]))].index)
RFtest2 = RFtest.drop(RFtest[(RFtest['Ramp'].isin([0]))].index)
SVRtest2 = SVRtest.drop(SVRtest[(SVRtest['Ramp'].isin([0]))].index)
ANNtest2 = ANNtest.drop(ANNtest[(ANNtest['Ramp'].isin([0]))].index)


# In[344]:


test_power2


# In[346]:


df_forecast2


# In[350]:



plt.plot(test_power2['Date'], test_power2['ActivePower'], color='grey',marker='.',alpha=0.7, label='true')
plt.plot(df_forecast2['Date'], df_forecast2['ActivePower'], color='red', marker='.',linestyle = "--",alpha=0.9, label='LSTM')

plt.xticks(rotation = '45')
plt.xlabel('Date'); plt.ylabel('ActivePower'); plt.title('Actual and Predicted Ramp Event Values');
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[219]:


plt.figure(figsize=(20,20))
plt.plot(test_power['Date'], test_power['ActivePower'], color='grey',marker='.',alpha=0.5, label='true')


plt.plot(df_forecast2['Date'], df_forecast2['ActivePower'], color='red', marker='.',linestyle = "--",alpha=0.7, label='LSTM')
plt.plot(RFtest2['date'], RFtest2['ActivePower'], color='blue',marker='.',linestyle = "--",alpha=0.7,label = 'RandomForest')
plt.plot(svcTestPred_data2['date'], svcTestPred_data2['ActivePower'], color='green',marker='.',linestyle = "--",alpha=0.7,label = 'SVM')
plt.plot(yp_data2['Date'], yp_data2['ActivePower'], color='m',marker='.',linestyle = "--",alpha=0.7,label = 'ANN')
plt.plot(new_predcml2['date'], new_predcml2['ActivePower'], color='yellow',marker='.',linestyle = "--",alpha=0.7,label = 'CMTL')

# Graph labels
plt.xticks(rotation = '45')
plt.xlabel('Date'); plt.ylabel('ActivePower'); plt.title('Actual and Predicted Values');
plt.legend(bbox_to_anchor)


# In[331]:


import sklearn.metrics as metrics


# In[ ]:


df_forecast = pd.DataFrame({'Date':train_dates, 'ActivePower':y_preddata})
original = df[['Date', 'ActivePower','r']]

original = original.loc[original['Date'] >= '2018-11-20 00:00:00']


# In[352]:


# calculate manually
def acc(y,yhat):
    
    d = y - yhat
    mse_f = np.mean(d**2)
    mae_f = np.mean(abs(d))
    rmse_f = np.sqrt(mse_f)
    r2_f = 1-(sum(d**2)/sum((y-np.mean(y))**2))

    print("Results by manual calculation:")
    print("MAE:",mae_f)
    print("MSE:", mse_f)
    print("RMSE:", rmse_f)
    print("R-Squared:", r2_f)
    list(r2_f)


# In[353]:


LSTMAC = acc(y_test,y_pred)


# In[338]:


LSTMR = 1-(sum((y_test - y_pred)**2)/sum((y_test-np.mean(y_test))**2))


# In[339]:


LSTMR


# In[332]:


RandomForestAC = acc(test_Y,yTestPred)


# In[334]:


SVMAC = acc(test_Y,svrTestPred)


# In[335]:


ANNac = acc(test_Y,ANNyp)


# In[277]:


Y_test


# In[278]:


ANNyp


# In[276]:


CMTLAC = acc(df['ActivePower'],predcml)


# In[319]:


df_forecast['ActivePower']


# In[320]:


test['WindSpeed']


# In[342]:


plt.hist(LSTMR, # 绘图数据
         bins = 100, # 指定直方图条的个数
         density=True, 
         stacked=True,
         color = 'steelblue', # 指定填充色
         edgecolor = 'k') # 指定直方图的边界色


# In[ ]:




