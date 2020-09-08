import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import random
import math

os.chdir("C:\\Users\\ahmed\\Desktop")

st.title("MARATHON TRAINING APP")
#title = st.text_input('Select Your Country', 'is')
#st.write('The current country selected is', title)
#currentDistance = st.slider('How long do you currently run for?', 5, 60, 10)
#st.write('Distance range considered: ', currentDistance-2, '-', currentDistance+2, 'KM')
#projectedDistance = st.slider('How many KMs is your aim?', 5, 60, 20)
df_original = pd.read_pickle('valid_2016.pkl')

#STEP 1
#We only want the marathon run times first
marathon = df_original.copy()
marathon.dropna(subset = ["time_final"], inplace=True)
#marathon

#Say we have a user, they specify what is their goal time.
Hours = st.number_input("Insert Hours", value = 3, min_value = int(0), max_value=5, step=1)
Minutes = st.number_input("Insert Minutes", value = 27, min_value = int(0), max_value=60, step=1)
Seconds = st.number_input("Insert Seconds", value = 0, min_value = int(0), max_value=60, step=1)
hours = 3
mins = 27
seconds = 0
time_wanted = (Hours*60*60) + (Minutes*60) + Seconds
#We will take anything within the 5 secs range
marathon_new = marathon[marathon.time_final < (time_wanted + 5)]
marathon_new = marathon_new[marathon_new.time_final > (time_wanted - 5)]


list_of_ids = marathon_new['hashedathleteid'].to_list()
if st.checkbox('Show ID List'):
    'ID List', list_of_ids
#STEP 1 DONE - Users Identified


#Get list of all activites of users found
lenofid = len(list_of_ids)
i = 0
DF_list= list()
while i<lenofid:
    temp = df_original[df_original['hashedathleteid'].str.contains(list_of_ids[i])]
    #print(temp.head(2))
    #new = pd.concat([end, temp])
    DF_list.append(temp)
    i += 1

activities_final = pd.concat(DF_list)  

#add a new column : average pcae
avg_pace = activities_final.pace_cumul.values

length = len(avg_pace) 
result = []   
 
for i in range(length): 
    result.append(avg_pace[i][-1])
    
activities_final['avg_pace'] = result


#add a new column : max pace
max_pace = activities_final.pace_diff.values

m_length = len(max_pace) 
m_result = []   
 
for i in range(m_length): 
    m_result.append(max(max_pace[i]))
    
activities_final['max_pace'] = m_result


#add a new column : min pace
def second_smallest(numbers):
    m1, m2 = float('inf'), float('inf')
    for x in numbers:
        if x <= m1:
            m1, m2 = x, m1
        elif x < m2:
            m2 = x
    return m2

min_pace = activities_final.pace_diff.values

mi_length = len(min_pace) 
mi_result = []   
 
for i in range(mi_length): 
    mi_result.append(second_smallest(min_pace[i]))
    
activities_final['min_pace'] = mi_result

#add a new column : stdv pace
pace = activities_final.pace_diff.values
s_length = len(pace)
s_result = []
for i in range(s_length): 
    s_result.append(np.std(pace[i]))

activities_final['stdv_pace'] = s_result

#add a new column : gap
times = pd.Series(activities_final["startdatelocal"])
y = np.diff(times)
results = []
for i in range(len(y)):
    x = y[i]
    hours = x.astype('timedelta64[h]')  #in hours
    results.append(hours / np.timedelta64(1, 'h'))

results = [None if i < 0 else i for i in results] #contains gap between excersise but need to be per runner
results.insert(0,None)
activities_final['gap/interval'] = results

if st.checkbox('All Activities'):
    'All Activities', activities_final

#Put these in a weekly format
dr = pd.date_range('2016-06-01 00:00:00', periods=32, freq='W-SUN')
def featureEngineering(df):
    gr = df.groupby(["hashedathleteid", pd.Grouper(key='startdatelocal',label='left', freq='W-SUN')])
    for (hashedathleteid, week), group in gr:
        print(hashedathleteid, week)
        total_km = group["totaldistance"].sum()/1000
        max_km = group["totaldistance"].max()/1000
        min_km = group["totaldistance"].min()/1000
        var_km = math.sqrt(group["totaldistance"].var())/1000
        average_km = group["totaldistance"].mean()/1000
        num_runs = group["totaldistance"].count()
        average_pace = group["avg_pace"].mean()
        max_pace = group["max_pace"].max()
        min_pace = group["min_pace"].min()
        stdv_pace = group["stdv_pace"].mean()
        gap = group["gap/interval"].mean()
#         date_label
        num_20k = group[group['totaldistance'] >= 20000]['totaldistance'].count()
        num_30k = group[group['totaldistance'] >= 30000]['totaldistance'].count()
        
        
        #print(group)
        yield {'hashedathleteid': hashedathleteid, 
               'week': week,
               'total_km': total_km, 'num_runs': num_runs, 'num_20k': num_20k, 'num_30k': num_30k, 
               'max_km': max_km, 'min_km': min_km, 'stdv_km': var_km,'average_km' : average_km,  
               'max_pace': max_pace, 'min_pace': min_pace, 'stdv_pace': stdv_pace, 'average_pace': average_pace, 'avg_gap': gap}
    return
a_id = list_of_ids
df_features = pd.DataFrame(featureEngineering(activities_final.query('hashedathleteid == @a_id')))
#df_features = df_features.set_index('week')
if st.checkbox('Weekly dataframe'):
    'Weekly Dataframe', df_features
#STEP 2 DONE
#df_features: contains a weekly summary for each runner
#activites_final: contains all activites done by the selected runners

st.line_chart(df_features)

df = pd.DataFrame(
     np.random.randn(200, 3),
     columns=['a', 'b', 'c'])

chart = alt.Chart(df_features).mark_line().encode(
    x='average_pace',
    y='total_km'
)
st.write(chart)
