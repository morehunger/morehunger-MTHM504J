#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta
from datetime import datetime
from itertools import groupby
import math


# In[2]:


## read the data 
wxj = pd.read_csv("September.csv")


# In[36]:


wxj['Date/Time'] = pd.to_datetime(wxj['Date/Time'], format ="%d %m %Y %H:%M")


# In[37]:


wxj


# In[5]:


cut = pd.read_csv("September.csv")
cut['Date/Time'] = pd.to_datetime(cut['Date/Time'], format ="%d %m %Y %H:%M")
cut


# In[6]:


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


# In[7]:


ramp = swinging_door(wxj.index,wxj['LV ActivePower (kW)'],wxj['Date/Time'],0.1)


# In[332]:


d = abs(wxj['LV ActivePower (kW)'].shift(-1) - wxj['LV ActivePower (kW)'])
d.replace([np.inf, -np.inf], np.nan, inplace=True)
d.replace(np.nan, 0, inplace=True)
d


# In[333]:


bossavyd = abs(wxj['LV ActivePower (kW)'] - wxj['LV ActivePower (kW)'].shift(+30))/30
bossavyd.replace([np.inf, -np.inf], np.nan, inplace=True)
bossavyd.replace(np.nan, 0, inplace=True)
bossavyd.dropna(axis=0, how='any', inplace=True)


# In[334]:


bossavyd


# In[335]:


cut.insert(cut.shape[1],"difference",abs(cut['LV ActivePower (kW)'] - cut['LV ActivePower (kW)'].shift(+18)))
cut


# In[336]:


def cutler1(data_x,data_y,data_z,data_m,data_n):
    
    cutler_x = []
    cutler_y = []
    cutler_z = []
    i=0

    for j in data_z:
            for number in data_m:
                if number > j*0.75:
                    cutler_x.append(data_x[i])
                    cutler_y.append(data_y[i])
                    cutler_z.append(data_n[i])
                    i = i+1
                else:
                    i = i+1
            return cutler_x,cutler_y,cutler_z   
    
            


# In[337]:


c= cutler1(cut.index,cut['LV ActivePower (kW)'],cut['Theoretical_Power_Curve (KWh)'],cut["difference"],cut['Date/Time'])
c


# In[338]:


for i in c[0]:
       cut.drop(cut.index[i],inplace=True)


# In[339]:


cut.reset_index(inplace=True)
cut


# In[340]:


cut = cut.rename(columns={'index': 'xuhao'})
cut


# In[341]:


cd = abs(cut['LV ActivePower (kW)'] - cut['LV ActivePower (kW)'].shift(+6))
cd.replace([np.inf, -np.inf], np.nan, inplace=True)
cd.replace(np.nan, 0, inplace=True)
cd


# In[342]:


cut.insert(cut.shape[1],"difference2",cd)


# In[343]:


cutweeks = [g for n, g in cut.groupby(pd.Grouper(key='Date/Time',freq='W'))]
for cut_of_the_week in cutweeks:
    
    display(cut_of_the_week)


# In[344]:


bd = wxj['LV ActivePower (kW)'] / wxj['Theoretical_Power_Curve (KWh)']
bdd = abs(bd - bd.shift(+1))
bdd.replace([np.inf, -np.inf], np.nan, inplace=True)
bdd.replace(np.nan, 0, inplace=True)
bdd


# In[345]:


wxj.insert(wxj.shape[1],"PPC difference",bdd)
wxj.insert(wxj.shape[1],"d",d)
wxj.insert(wxj.shape[1],"bossavyd",bossavyd)


# In[346]:


weeks = [g for n, g in wxj.groupby(pd.Grouper(key='Date/Time',freq='W'))]
for wxj_of_the_week in weeks:
    
    display(wxj_of_the_week)


# In[347]:


def cutler2(data_x,data_y,data_z,data_m,data_n):
    
    cutler_x = []
    cutler_y = []
    cutler_z = []
    i=0

    for j in data_z:
            for number in data_m:
                if number > j*0.65:
                    cutler_x.append(data_x[i])
                    cutler_y.append(data_y[i])
                    cutler_z.append(data_n[i])
                    i = i+1
                else:
                    i = i+1
            return cutler_x,cutler_y,cutler_z   
    
            


# In[348]:


def greaves(data_x,data_y,data_z,data_m,data_n):
    
    cutler_x = []
    cutler_y = []
    cutler_z = []
    i=0

    for number in data_m:
            for j in data_z:
                if number > j*0.5:
                    cutler_x.append(data_x[i])
                    cutler_y.append(data_y[i])
                    cutler_z.append(data_n[i])
                    i = i+1
                    
                else:
                    i = i+1
                    
            return cutler_x,cutler_y,cutler_z
            


# In[356]:


def bossavy(data_x,data_y,data_z,data_m,data_n):
    
    cutler_x = []
    cutler_y = []
    cutler_z = []
    i=0

    for number in data_m:
            for j in data_z:
                if number < j*0.25:
                    
                    i = i+1
                else:
                    cutler_x.append(data_x[i])
                    cutler_y.append(data_y[i])
                    cutler_z.append(data_n[i])
                    i=i+1
                    
            return cutler_x,cutler_y,cutler_z
            


# In[350]:


def bradford(data_x,data_y,data_z,data_n):
    
    cutler_x = []
    cutler_y = []
    cutler_z = []
    i=0
    for j in data_n:
            if j > 0.2:
                cutler_x.append(data_x[i])
                cutler_y.append(data_y[i])
                cutler_z.append(data_z[i])
                i = i+1
                
    return cutler_x,cutler_y,cutler_z
            


# In[357]:


c= cutler2(cut['xuhao'],cut['LV ActivePower (kW)'],cut['Theoretical_Power_Curve (KWh)'],cut['difference2'],cut['Date/Time'])
g= greaves(wxj.index,wxj['LV ActivePower (kW)'],wxj['Theoretical_Power_Curve (KWh)'],wxj['d'],wxj['Date/Time'])
bo= bossavy(wxj.index,wxj['LV ActivePower (kW)'],wxj['Theoretical_Power_Curve (KWh)'],wxj['bossavyd'],wxj['Date/Time'])
br= bradford(wxj.index,wxj['LV ActivePower (kW)'],wxj['Date/Time'],wxj['PPC difference'])


# In[358]:


bo


# In[359]:


wxj['r']=0
wxj['c']=0  
wxj['g']=0
wxj['bo']=0
wxj['br']=0


# In[360]:


for i in ramp[0]:
       for j in list(wxj.index):
            
            if i==j:
                wxj.loc[j,"r"]=10
                break


# In[361]:


for i in c[0]:
       for j in list(wxj.index):
            
            if i==j:
                wxj.loc[j,"c"]=10
                break


# In[362]:


for i in g[0]:
       for j in list(wxj.index):
            
            if i==j:
                wxj.loc[j,"g"]=10
                break


# In[363]:


for i in bo[0]:
       for j in list(wxj.index):
            
            if i==j:
                wxj.loc[j,"bo"]=10
                break


# In[364]:


for i in br[0]:
       for j in list(wxj.index):
            
            if i==j:
                wxj.loc[j,"br"]=10
                break


# In[365]:


ramp1 = swinging_door(weeks[0].index,weeks[0]['LV ActivePower (kW)'],weeks[0]['Date/Time'],0.3)


# In[366]:


weeks[1].index=(weeks[1].index-288)
ramp2 = swinging_door(weeks[1].index,weeks[1]['LV ActivePower (kW)'],weeks[1]['Date/Time'],0.3)


# In[367]:


weeks[2].index=(weeks[2].index-1295)
ramp3 = swinging_door(weeks[2].index,weeks[2]['LV ActivePower (kW)'],weeks[2]['Date/Time'],0.3)


# In[368]:


weeks[3].index=(weeks[3].index-2287)
ramp4 = swinging_door(weeks[3].index,weeks[3]['LV ActivePower (kW)'],weeks[3]['Date/Time'],0.3)


# In[369]:


weeks[4].index=(weeks[4].index-3295)
ramp5 = swinging_door(weeks[4].index,weeks[4]['LV ActivePower (kW)'],weeks[4]['Date/Time'],0.3)


# In[371]:


cutweeks[1].index=(cutweeks[1].index-288)
cutweeks[2].index=(cutweeks[2].index-1293)
cutweeks[3].index=(cutweeks[3].index-2266)
cutweeks[4].index=(cutweeks[4].index-3274)


# In[372]:


c1= cutler2(cutweeks[0]['xuhao'],cutweeks[0]['LV ActivePower (kW)'],cutweeks[0]['Theoretical_Power_Curve (KWh)'],cutweeks[0]['difference2'],cutweeks[0]['Date/Time'])
g1= greaves(weeks[0].index,weeks[0]['LV ActivePower (kW)'],weeks[0]['Theoretical_Power_Curve (KWh)'],weeks[0]['d'],weeks[0]['Date/Time'])
bo1= bossavy(weeks[0].index,weeks[0]['LV ActivePower (kW)'],weeks[0]['Theoretical_Power_Curve (KWh)'],weeks[0]['bossavyd'],weeks[0]['Date/Time'])
br1= bradford(weeks[0].index,weeks[0]['LV ActivePower (kW)'],weeks[0]['Date/Time'],weeks[0]['PPC difference'])


# In[373]:


c2= cutler2(cutweeks[1]['xuhao'],cutweeks[1]['LV ActivePower (kW)'],cutweeks[1]['Theoretical_Power_Curve (KWh)'],cutweeks[1]['difference2'],cutweeks[1]['Date/Time'])
g2= greaves(weeks[1].index,weeks[1]['LV ActivePower (kW)'],weeks[1]['Theoretical_Power_Curve (KWh)'],weeks[1]['d'],weeks[1]['Date/Time'])
bo2= bossavy(weeks[1].index,weeks[1]['LV ActivePower (kW)'],weeks[1]['Theoretical_Power_Curve (KWh)'],weeks[1]['bossavyd'],weeks[1]['Date/Time'])
br2= bradford(weeks[1].index,weeks[1]['LV ActivePower (kW)'],weeks[1]['Date/Time'],weeks[1]['PPC difference'])


# In[374]:


c3= cutler2(cutweeks[2]['xuhao'],cutweeks[2]['LV ActivePower (kW)'],cutweeks[2]['Theoretical_Power_Curve (KWh)'],cutweeks[2]['difference2'],cutweeks[2]['Date/Time'])
g3= greaves(weeks[2].index,weeks[2]['LV ActivePower (kW)'],weeks[2]['Theoretical_Power_Curve (KWh)'],weeks[2]['d'],weeks[2]['Date/Time'])
bo3= bossavy(weeks[2].index,weeks[2]['LV ActivePower (kW)'],weeks[2]['Theoretical_Power_Curve (KWh)'],weeks[2]['bossavyd'],weeks[2]['Date/Time'])
br3= bradford(weeks[2].index,weeks[2]['LV ActivePower (kW)'],weeks[2]['Date/Time'],weeks[2]['PPC difference'])


# In[375]:


c4= cutler2(cutweeks[3]['xuhao'],cutweeks[3]['LV ActivePower (kW)'],cutweeks[3]['Theoretical_Power_Curve (KWh)'],cutweeks[3]['difference2'],cutweeks[3]['Date/Time'])
g4= greaves(weeks[3].index,weeks[3]['LV ActivePower (kW)'],weeks[3]['Theoretical_Power_Curve (KWh)'],weeks[3]['d'],weeks[3]['Date/Time'])
bo4= bossavy(weeks[3].index,weeks[3]['LV ActivePower (kW)'],weeks[3]['Theoretical_Power_Curve (KWh)'],weeks[3]['bossavyd'],weeks[3]['Date/Time'])
br4= bradford(weeks[3].index,weeks[3]['LV ActivePower (kW)'],weeks[3]['Date/Time'],weeks[3]['PPC difference'])


# In[376]:


c5= cutler2(cutweeks[4]['xuhao'],cutweeks[4]['LV ActivePower (kW)'],cutweeks[4]['Theoretical_Power_Curve (KWh)'],cutweeks[4]['difference2'],cutweeks[4]['Date/Time'])
g5= greaves(weeks[4].index,weeks[4]['LV ActivePower (kW)'],weeks[4]['Theoretical_Power_Curve (KWh)'],weeks[4]['d'],weeks[4]['Date/Time'])
bo5= bossavy(weeks[4].index,weeks[4]['LV ActivePower (kW)'],weeks[4]['Theoretical_Power_Curve (KWh)'],weeks[4]['bossavyd'],weeks[4]['Date/Time'])
br5= bradford(weeks[4].index,weeks[4]['LV ActivePower (kW)'],weeks[4]['Date/Time'],weeks[4]['PPC difference'])


# In[379]:


plt.figure(figsize=(20,20))

ax1 = plt.subplot(611) 
plt.plot(wxj['Date/Time'],wxj['LV ActivePower (kW)'], color='grey',label='Wind power')
plt.plot(ramp[2],ramp[1],color='red', linestyle = "--",label='SDA')
plt.plot(c[2],c[1], color='blue', linestyle = "--",label='cutler')
plt.plot(g[2],g[1], linestyle="--", color='green',label='greaves')
plt.plot(bo[2],bo[1], linestyle="--", color='m',label='bossavy')
plt.plot(br[2],br[1], linestyle="--", color='yellow',label='bradford')

ax2 = plt.subplot(612)
plt.scatter(wxj['Date/Time'],wxj['r'], marker='o',s=4.0, color='grey',label='SDA')
plt.plot(wxj['Date/Time'],wxj['r'], linestyle='solid', color='grey',alpha=0.3)

ax3 = plt.subplot(613)
plt.scatter(wxj['Date/Time'],wxj['c'], marker='o',s=4.0, color='grey',label='cutler')
plt.plot(wxj['Date/Time'],wxj['c'], linestyle='solid', color='grey',alpha=0.3)

ax4 = plt.subplot(614)
plt.scatter(wxj['Date/Time'],wxj['g'], marker='o',s=4.0, color='grey',label='greaves')
plt.plot(wxj['Date/Time'],wxj['g'], linestyle='solid', color='grey',alpha=0.3)

ax5 = plt.subplot(615)
plt.scatter(wxj['Date/Time'],wxj['bo'], marker='o',s=4.0, color='grey',label='bossavy')
plt.plot(wxj['Date/Time'],wxj['bo'], linestyle='solid', color='grey',alpha=0.3)

ax6 = plt.subplot(616)
plt.scatter(wxj['Date/Time'],wxj['br'], marker='o',s=4.0, color='grey',label='bradford')
plt.plot(wxj['Date/Time'],wxj['br'], linestyle='solid', color='grey',alpha=0.3)

ax1.legend()         
ax2.legend()         
ax3.legend()
ax4.legend()         
ax5.legend()         
ax6.legend()
plt.show()


# In[380]:


plt.figure(figsize=(20,20))

ax1 = plt.subplot(611) 
plt.plot(wxj['Date/Time'],wxj['LV ActivePower (kW)'], color='grey',label='Wind power')
plt.plot(ramp[2],ramp[1],color='red', linestyle = "--",label='SDA')
plt.plot(c[2],c[1], color='blue', linestyle = "--",label='cutler')
plt.plot(g[2],g[1], linestyle="--", color='green',label='greaves')
plt.plot(bo[2],bo[1], linestyle="--", color='m',label='bossavy')
plt.plot(br[2],br[1], linestyle="--", color='yellow',label='bradford')

ax2 = plt.subplot(612)
plt.plot(weeks[0]['Date/Time'],weeks[0]['LV ActivePower (kW)'], color='grey',label='Wind power')
plt.plot(ramp1[2],ramp1[1],color='red', linestyle = "--",label='SDA')
plt.plot(c1[2],c1[1], color='blue', linestyle = "--",label='cutler')
plt.plot(g1[2],g1[1], linestyle="--", color='green',label='greaves')
plt.plot(bo1[2],bo1[1], linestyle="--", color='m',label='bossavy')
plt.plot(br1[2],br1[1], linestyle="--", color='yellow',label='bradford')

ax3 = plt.subplot(613)
plt.plot(weeks[1]['Date/Time'],weeks[1]['LV ActivePower (kW)'], color='grey',label='Wind power')
plt.plot(ramp2[2],ramp2[1],color='red', linestyle = "--",label='SDA')
plt.plot(c2[2],c2[1], color='blue', linestyle = "--",label='cutler')
plt.plot(g2[2],g2[1], linestyle="--", color='green',label='greaves')
plt.plot(bo2[2],bo2[1], linestyle="--", color='m',label='bossavy')
plt.plot(br2[2],br2[1], linestyle="--", color='yellow',label='bradford')

ax4 = plt.subplot(614)
plt.plot(weeks[2]['Date/Time'],weeks[2]['LV ActivePower (kW)'], color='grey',label='Wind power')
plt.plot(ramp3[2],ramp3[1],color='red', linestyle = "--",label='SDA')
plt.plot(c3[2],c3[1], color='blue', linestyle = "--",label='cutler')
plt.plot(g3[2],g3[1], linestyle="--", color='green',label='greaves')
plt.plot(bo3[2],bo3[1], linestyle="--", color='m',label='bossavy')
plt.plot(br3[2],br3[1], linestyle="--", color='yellow',label='bradford')

ax5 = plt.subplot(615)
plt.plot(weeks[3]['Date/Time'],weeks[3]['LV ActivePower (kW)'], color='grey',label='Wind power')
plt.plot(ramp4[2],ramp4[1],color='red', linestyle = "--",label='SDA')
plt.plot(c4[2],c4[1], color='blue', linestyle = "--",label='cutler')
plt.plot(g4[2],g4[1], linestyle="--", color='green',label='greaves')
plt.plot(bo4[2],bo4[1], linestyle="--", color='m',label='bossavy')
plt.plot(br4[2],br4[1], linestyle="--", color='yellow',label='bradford')
         
ax6 = plt.subplot(616)
plt.plot(weeks[4]['Date/Time'],weeks[4]['LV ActivePower (kW)'], color='grey',label='Wind power')
plt.plot(ramp5[2],ramp5[1],color='red', linestyle = "--",label='SDA')
plt.plot(c5[2],c5[1], color='blue', linestyle = "--",label='cutler')
plt.plot(g5[2],g5[1], linestyle="--", color='green',label='greaves')
plt.plot(bo5[2],bo5[1], linestyle="--", color='m',label='bossavy')
plt.plot(br5[2],br5[1], linestyle="--", color='yellow',label='bradford') 

ax1.legend()         
ax2.legend()         
ax3.legend()
ax4.legend()         
ax5.legend()         
ax6.legend()         
plt.show()


# In[ ]:




