from math import sqrt
import streamlit as st
import altair as alt
import pydeck as pdk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import time
import seaborn as sns


os.chdir("C:\\Users\\ahmed\\Desktop")
df_original = pd.read_pickle('valid_2016.pkl')
st.title("MARATHON TRAINING APP")
st.write("This app analyses your training pattern and asks you for your marathon goal time. Your training data is then compared to the training data of runners who achieved your specified goal time and have a similar training pattern to you.")


# STEP 1 - We only want the marathon run times first
marathon = df_original.copy()
marathon.dropna(subset=["time_final"], inplace=True)

wanted_user = marathon.head(1)

# get all their activities
requiredid = wanted_user['hashedathleteid'].to_list()  # get the id
wanted_user_activities = df_original[df_original['hashedathleteid'].str.contains(
    requiredid[0])].copy()

# add needed columns
# add a new column : average pcae
avg_pace = wanted_user_activities.pace_cumul.values

length = len(avg_pace)
result = []

for i in range(length):
    result.append(avg_pace[i][-1])

wanted_user_activities['avg_pace'] = result

# add a new column : max pace
max_pace = wanted_user_activities.pace_diff.values

m_length = len(max_pace)
m_result = []

for i in range(m_length):
    m_result.append(max(max_pace[i]))

wanted_user_activities['max_pace'] = m_result


# add a new column : min pace
def second_smallest(numbers):
    m1, m2 = float('inf'), float('inf')
    for x in numbers:
        if x <= m1:
            m1, m2 = x, m1
        elif x < m2:
            m2 = x
    return m2


min_pace = wanted_user_activities.pace_diff.values

mi_length = len(min_pace)
mi_result = []

for i in range(mi_length):
    mi_result.append(second_smallest(min_pace[i]))

wanted_user_activities['min_pace'] = mi_result

# add a new column : stdv pace
pace = wanted_user_activities.pace_diff.values
s_length = len(pace)
s_result = []
for i in range(s_length):
    s_result.append(np.std(pace[i]))

wanted_user_activities['stdv_pace'] = s_result

# add a new column : gap
times = pd.Series(wanted_user_activities["startdatelocal"])
y = np.diff(times)
results = []
for i in range(len(y)):
    x = y[i]
    hours = x.astype('timedelta64[h]')  # in hours
    results.append(hours / np.timedelta64(1, 'h'))

# contains gap between excersise but need to be per runner
results = [None if i < 0 else i for i in results]
results.insert(0, None)
wanted_user_activities['gap/interval'] = results

# get weekly activities
dr = pd.date_range('2016-06-01 00:00:00', periods=32, freq='W-SUN')


def featureEngineering(df):
    gr = df.groupby(["hashedathleteid", pd.Grouper(
        key='startdatelocal', label='left', freq='W-SUN')])
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
        num_20k = group[group['totaldistance']
                        >= 20000]['totaldistance'].count()
        num_30k = group[group['totaldistance']
                        >= 30000]['totaldistance'].count()

        # print(group)
        yield {'hashedathleteid': hashedathleteid,
               'week': week,
               'total_km': total_km, 'num_runs': num_runs, 'num_20k': num_20k, 'num_30k': num_30k,
               'max_km': max_km, 'min_km': min_km, 'stdv_km': var_km, 'average_km': average_km,
               'max_pace': max_pace, 'min_pace': min_pace, 'stdv_pace': stdv_pace, 'average_pace': average_pace, 'gap': gap}
    return


a_id = requiredid
weekly_user_df = pd.DataFrame(featureEngineering(
    wanted_user_activities.query('hashedathleteid == @a_id')))
st.subheader("\tYOUR STATISTICS")
st.subheader("Your Distances")
st.write("These are your weekly distance statistics. __Kilometers vs Week__.")

graphed_weekly_user1 = weekly_user_df[[
    'week', 'total_km', 'max_km', 'min_km', 'average_km']]
graphed_weekly_user1.set_index("week", inplace=True)
st.line_chart(graphed_weekly_user1)
st.subheader("Your Pace")
st.write("These are your weekly pace statistics. __Miles per hour vs Week__.")
graphed_weekly_user2 = weekly_user_df[[
    'week', 'max_pace', 'min_pace', 'average_pace']]
graphed_weekly_user2.set_index("week", inplace=True)
st.line_chart(graphed_weekly_user2)
st.subheader("Your Runs")
st.write("These are your types of runs weekly specifying if they are under 20KM, 30KM or more. __Number of runs vs Week__.")
graphed_weekly_user3 = weekly_user_df[[
    'week', 'num_runs', 'num_20k', 'num_30k']]
graphed_weekly_user3['under 20k'] = graphed_weekly_user3['num_runs'] - \
    graphed_weekly_user3['num_20k'] - graphed_weekly_user3['num_30k']
graphed_weekly_user3 = graphed_weekly_user3.drop(columns=['num_runs'])
#graphed_weekly_user3['weeks'] = np.arange(len(graphed_weekly_user3))
graphed_weekly_user3.set_index("week", inplace=True)
graphed_weekly_user3 = graphed_weekly_user3.rename(
    columns={"num_20k": "20k - 30k", "num_30k": "30k +"})
st.bar_chart(graphed_weekly_user3)
#sns.barplot(x='weeks', y="num_runs", hue="num_30k", data=graphed_weekly_user3)
# st.pyplot()
st.subheader("Average Gap between runs")
st.write("These are your Average Weekly Gap between runs. __Hours vs Week__.")
graphed_weekly_user4 = weekly_user_df[['week', 'gap']]
graphed_weekly_user4.set_index("week", inplace=True)
st.line_chart(graphed_weekly_user4)
exp_time = time.strftime('%H:%M:%S', time.gmtime(wanted_user['time_final']))
st.write("Your expected **finish** time:", exp_time)

# User specify goal time
st.subheader("-  \nSpecify your Goal time")
Hours = st.number_input("Insert Hours", value=3,
                        min_value=int(0), max_value=5, step=1)
Minutes = st.number_input("Insert Minutes", value=27,
                          min_value=int(0), max_value=60, step=1)
Seconds = st.number_input("Insert Seconds", value=0,
                          min_value=int(0), max_value=60, step=1)
time_wanted = (Hours*60*60) + (Minutes*60) + Seconds
# We will take anything within the 5 mins range (STEP 1 OF FILTRING)
marathon_new = marathon[marathon.time_final < (time_wanted + 300)]
marathon_new = marathon_new[marathon_new.time_final > (time_wanted - 300)]
list_of_ids = marathon_new['hashedathleteid'].to_list()
# STEP 1 of filtiring DONE - Users Identified

# Get list of all activites of users found
#Get list of all activites of users found
#end = final
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
avg_pace2 = activities_final.pace_cumul.values

for jkj in range(len(avg_pace2)):
    if (len(avg_pace2[jkj]) == 0):
        avg_pace2[jkj].append(None) 
        #avg_pace2 = avg_pace2.delete(avg_pace2, jkj, 0)


length2 = len(avg_pace2) 
result2 = []   



for kkk in range(length2):
    #print(kkk)
    result2.append(avg_pace2[kkk][-1])
    
activities_final['avg_pace'] = result2

#add a new column : max pace
max_pace = activities_final.pace_diff.values
for jkj2 in range(len(max_pace)):
    if (len(max_pace[jkj2]) == 0):
        max_pace[jkj2].append(None) 
m_length = len(max_pace) 
m_result = []   
 
for iii in range(m_length): 
    m_result.append(max(max_pace[iii]))
    
activities_final['max_pace'] = m_result


#add a new column : min pace
def second_smallest(numbers):    
    m1, m2 = float('inf'), float('inf')
    if numbers[0] == None:
        return 0
    if m1 == None:
        return m2
    if m2 == None:
        return m1
    else:
        for x in numbers:
            if x <= m1:
                m1, m2 = x, m1
            elif x < m2:
                m2 = x
        return m2

min_pace = activities_final.pace_diff.values

mi_length = len(min_pace) 
mi_result = []   
 
for i2 in range(mi_length): 
    mi_result.append(second_smallest(min_pace[i2]))
    
activities_final['min_pace'] = mi_result


import statistics
#add a new column : stdv pace
pace = activities_final.pace_diff.values

s_length = len(pace)
s_result = []
for i3 in range(s_length):
    if len(pace[i3]) < 2:
        s_result.append(float(0))
    else:
        s_result.append(statistics.stdev(pace[i3]))

activities_final['stdv_pace'] = s_result

#add a new column : gap
times = pd.Series(activities_final["startdatelocal"])
y = np.diff(times)
results = []
for i4 in range(len(y)):
    x = y[i4]
    hours = x.astype('timedelta64[h]')  #in hours
    results.append(hours / np.timedelta64(1, 'h'))

results = [None if i4 < 0 else i4 for i4 in results] #contains gap between excersise but need to be per runner
results.insert(0,None)
activities_final['gap/interval'] = results


# Put these in a weekly format
dr = pd.date_range('2016-06-01 00:00:00', periods=32, freq='W-SUN')


def featureEngineering(df):
    gr = df.groupby(["hashedathleteid", pd.Grouper(
        key='startdatelocal', label='left', freq='W-SUN')])
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
        num_20k = group[group['totaldistance']
                        >= 20000]['totaldistance'].count()
        num_30k = group[group['totaldistance']
                        >= 30000]['totaldistance'].count()

        # print(group)
        yield {'hashedathleteid': hashedathleteid,
               'week': week,
               'total_km': total_km, 'num_runs': num_runs, 'num_20k': num_20k, 'num_30k': num_30k,
               'max_km': max_km, 'min_km': min_km, 'stdv_km': var_km, 'average_km': average_km,
               'max_pace': max_pace, 'min_pace': min_pace, 'stdv_pace': stdv_pace, 'average_pace': average_pace, 'gap': gap}
    return


a_id = list_of_ids
df_features = pd.DataFrame(featureEngineering(
    activities_final.query('hashedathleteid == @a_id')))
# STEP 2 DONE
# df_features: contains a weekly summary for each runner
# activites_final: contains all activites done by the selected runners

# Put these in a yearly format
dr = pd.date_range('2016-06-01 00:00:00', periods=32, freq='Y')


def featureEngineering(df):
    gr = df.groupby(["hashedathleteid", pd.Grouper(
        key='startdatelocal', label='left', freq='Y')])
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
        num_20k = group[group['totaldistance']
                        >= 20000]['totaldistance'].count()
        num_30k = group[group['totaldistance']
                        >= 30000]['totaldistance'].count()

        # print(group)
        yield {'hashedathleteid': hashedathleteid,
               'year': week,
               'total_km': total_km, 'num_runs': num_runs, 'num_20k': num_20k, 'num_30k': num_30k,
               'max_km': max_km, 'min_km': min_km, 'stdv_km': var_km, 'average_km': average_km,
               'max_pace': max_pace, 'min_pace': min_pace, 'stdv_pace': stdv_pace, 'average_pace': average_pace, 'gap': gap}
    return


a_id = list_of_ids
year_df_features = pd.DataFrame(featureEngineering(
    activities_final.query('hashedathleteid == @a_id')))
# STEP 2 DONE
# df_features: contains a weekly summary for each runner
# activites_final: contains all activites done by the selected runners

# get yEARLY activities of selected user
dr = pd.date_range('2016-06-01 00:00:00', periods=32, freq='Y')


def featureEngineering(df):
    gr = df.groupby(["hashedathleteid", pd.Grouper(
        key='startdatelocal', label='left', freq='Y')])
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
        num_20k = group[group['totaldistance']
                        >= 20000]['totaldistance'].count()
        num_30k = group[group['totaldistance']
                        >= 30000]['totaldistance'].count()

        # print(group)
        yield {'hashedathleteid': hashedathleteid,
               'year': week,
               'total_km': total_km, 'num_runs': num_runs, 'num_20k': num_20k, 'num_30k': num_30k,
               'max_km': max_km, 'min_km': min_km, 'stdv_km': var_km, 'average_km': average_km,
               'max_pace': max_pace, 'min_pace': min_pace, 'stdv_pace': stdv_pace, 'average_pace': average_pace, 'gap': gap}
    return


a_id = requiredid
year_user_df = pd.DataFrame(featureEngineering(
    wanted_user_activities.query('hashedathleteid == @a_id')))

# KNN(Step 2 of filtirng)

from math import sqrt
#year_df_features
# calculate the Euclidean distance between two vectors
sami = year_df_features[['num_runs', 'gap']]
row0A = year_user_df[['num_runs','gap']]

#Scaling
w = ( row0A.gap - min(sami.gap) ) / ( max(sami.gap) - min(sami.gap) )
w2 = ( row0A.num_runs - min(sami.num_runs )) / ( max(sami.num_runs) - min(sami.num_runs) ) 
row0= pd.DataFrame()
row0["num_runs"] = w2
row0["gap"] = w

samiGap = []
samiRuns = []

for k in range(len(sami)):
    z = ( sami.loc[k, "gap"] - min(sami.gap) ) / ( max(sami.gap) - min(sami.gap) )
    z2 = ( sami.loc[k, "num_runs"] - min(sami.num_runs )) / ( max(sami.num_runs) - min(sami.num_runs) )
    samiGap.append(z)
    samiRuns.append(z2)

neigh = pd.DataFrame()
neigh["num_runs"] = samiRuns
neigh["gap"] = samiGap


def euclidean_distance(row1, row2):
    distance = (float(row1['gap']) - float(row2['gap']))**2 + (float(row1['num_runs']) - float(row2['num_runs']))**2
    return sqrt(distance)

ed = []

for j in range(len(neigh)):
    distance = euclidean_distance(row0, neigh.loc[j:j])
    ed.append(distance)

year_df_features["ed"] = ed
year_df_features = year_df_features.sort_values('ed')
#take the N nearest neighbours
if len(year_df_features) > 9:
    n = (len(year_df_features)*0.25)
else: 
    n = len(year_df_features)

n = int(n)
nearest_neighbours = year_df_features.head(n)
nearest_neighbours_id = nearest_neighbours["hashedathleteid"].to_list()

#year_df_features

#st.write("Number of nearest neighbours (runners):", n)

# Get the nearest neighbours weekly stats
number_of_neighbours = len(nearest_neighbours_id)
k = 0
knn_list = list()
while k < number_of_neighbours:
    tempo = df_features[df_features['hashedathleteid'].str.contains(
        nearest_neighbours_id[k])]
    knn_list.append(tempo)
    k += 1

neighbours_weekly = pd.concat(knn_list)

# neighbours_weekly
# nearest_neighbours
# year_user_df
# weekly_user_df
st.write("Neighbours weekly")
#st.write("Your Distances")
#neighbours_graph_weekly1 = neighbours_weekly[['week','total_km', 'max_km', 'min_km', 'average_km']]
#neighbours_graph_weekly1.set_index("week", inplace = True)
# st.bar_chart(neighbours_graph_weekly1)
# st.line_chart(neighbours_weekly)

######  GRAPHING        ###
neighbours_weekly['hashedathleteid'].value_counts()
weekly_user_df['hashedathleteid'].value_counts()

wl = len(weekly_user_df)

out = dict(tuple(neighbours_weekly.groupby("hashedathleteid")))
# len(out)
for ld in range(len(out)):
    if len(out[nearest_neighbours_id[ld]]) < wl:
        del out[nearest_neighbours_id[ld]]
    elif len(out[nearest_neighbours_id[ld]]) > wl:
        del out[nearest_neighbours_id[ld]]

# update knn
nearest_neighbours_id = list(out.keys())

st.write("Number of nearest neighbours (runners):", len(nearest_neighbours_id))

for j in range(len(nearest_neighbours_id)):
    out[nearest_neighbours_id[j]
        ] = out[nearest_neighbours_id[j]].reset_index(drop=True)

# Graph1
sum = 0

for i in range(len(nearest_neighbours_id)):
    sum += out[nearest_neighbours_id[i]].average_pace

average = sum/(i+1)
frame = {'average_pace': average}
all_avg = pd.DataFrame(frame)
all_avg['hashedathleteid'] = 'Other Runners'
weekly_user_df['hashedathleteid'] = 'You'
all_avg['weeks'] = np.arange(len(all_avg))
weekly_user_df['weeks'] = np.arange(len(weekly_user_df))
# all_avg
graph1 = pd.concat([weekly_user_df, all_avg], axis=0, ignore_index=True)

g = sns.lmplot(x="weeks", y="average_pace", hue="hashedathleteid",
               height=5, data=graph1, x_ci="sd")

# Use more informative axis labels than are provided by default
g.set_axis_labels("Week", "Average Pace")

st.text("\n");st.text("\n");st.text("\n");st.text("\n");st.text("\n")
st.title("COMPARING TO NEIGHBOUR RUNNERS")
st.write("Neighbour Runners are runners who ran the marathon within 5 minutes of your goal time & have a similar training style to you based on the weekly number of runs and gap between them.")
st.write("In the graphs below the points represent actual data points, the line represents the mean & the shaded area highlights the standard deviation.")

st.subheader("Average Pace")
st.write("This compares your weekly average pace (mile per hour) with the weekly averege pace of your neighbour runners. The higher the week number the closer to the marathon day.")
st.pyplot()

runnersMeanPace = round(all_avg.average_pace.mean(),2)
yourMeanPace = round(year_user_df.loc[0,"average_pace"],2)
st.write("Your average pace is: ", yourMeanPace, "mph, with standard deviation: ", round(weekly_user_df.average_pace.std(),2))
st.write("Other runners average pace is: ", runnersMeanPace, "mph, with standard deviation: ", round(all_avg.average_pace.std(),2))

if ((weekly_user_df.average_pace.std()- 0.5) > all_avg.average_pace.std()) or ((weekly_user_df.average_pace.std()+0.5) < all_avg.average_pace.std()):
    st.write("»Consider running at a more consistent pace so your body adjusts to it")
if yourMeanPace < runnersMeanPace:
    st.write("»Consider running faster for better perfomance")  
if yourMeanPace > runnersMeanPace:
    st.write("»Consider running slower to stop physical fatigue by pacing yourself")

##
dsum = 0

for d in range(len(nearest_neighbours_id)):
    dsum += out[nearest_neighbours_id[d]].average_km

daverage = dsum/(d+1)

d_frame = {'average_km': daverage}
d_avg = pd.DataFrame(d_frame)

d_avg['hashedathleteid'] = 'Other Runners'
d_avg['weeks'] = np.arange(len(d_avg))
graph2 = pd.concat([weekly_user_df, d_avg], axis=0, ignore_index=True)

g2 = sns.lmplot(x="weeks", y="average_km", hue="hashedathleteid",
                height=5, data=graph2, x_ci="sd")

# Use more informative axis labels than are provided by default
g2.set_axis_labels("Week", "Average KM")
st.subheader("Average Distance")
st.write("This compares your weekly average distance covered in a single run (Kilometers) with the weekly average distance of your neighbour runners. The higher the week number the closer to the marathon day.")
st.pyplot()

runnersMean = round(d_avg.average_km.mean(),2)
yourMean = round(year_user_df.loc[0, "average_km"],2)
st.write("Your average activity is: ", yourMean, "KM, with standard deviation: ", round(weekly_user_df.average_km.std(),2))
st.write("Other runners average activity is: ", runnersMean, "KM, with standard deviation: ", round(d_avg.average_km.std(),2))

if ((weekly_user_df.average_km.std()- 1) > d_avg.average_km.std()) or ((weekly_user_df.average_km.std()+1) < d_avg.average_km.std()):
    st.write("»Consider running more consistent distances so your body adjusts to it")
if yourMean < runnersMean:
    st.write("»Consider running longer distances to enhance endurance")  
if yourMean > runnersMean:
    st.write("»Consider running shorter distances to stop physical fatigue")
##
tdsum = 0

for td in range(len(nearest_neighbours_id)):
    tdsum += out[nearest_neighbours_id[td]].total_km

tdaverage = tdsum/(td+1)

td_frame = {'total_km': tdaverage}
td_avg = pd.DataFrame(td_frame)

td_avg['hashedathleteid'] = 'Other Runners'
td_avg['weeks'] = np.arange(len(td_avg))
graph3 = pd.concat([weekly_user_df, td_avg], axis=0, ignore_index=True)

g3 = sns.lmplot(x="weeks", y="total_km", hue="hashedathleteid",
                height=5, data=graph3, x_ci="sd")

# Use more informative axis labels than are provided by default
g3.set_axis_labels("Week", "Total KM")

st.subheader("Total Distance")
st.write("This compares your weekly total distance covered in all the runs (Kilometers) with the weekly total distance covered of your neighbour runners. The higher the week number the closer to the marathon day.")
st.pyplot()

runnersTD = round(td_avg.total_km.mean(), 2)
yourTD = round(weekly_user_df.total_km.mean(), 2)
st.write("Your weekly total distance is: ", yourTD, "with standard deviation: ", round(weekly_user_df.total_km.std(),2))
st.write("Other runners weekly total distance is: ", runnersTD, "with standard deviation: ", round(td_avg.total_km.std(),2))
st.write("Your overall total distance:",round(year_user_df.loc[0,"total_km"],2), "  Other runners overall total distance:", round(td_avg.total_km.sum(),2))


if ((weekly_user_df.total_km.std()- 3) > td_avg.total_km.std()) or ((weekly_user_df.total_km.std()+3) < td_avg.total_km.std()):
    st.write("»Consider running more consistent distances weekly so your body adjusts to it")
if yourTD < runnersTD:
    st.write("»Consider running longer distances weekly")  
if yourTD > runnersTD:
    st.write("»Consider running shorter distances weekly to stop physical fatigue by pacing yourself")
if year_user_df.loc[0,"total_km"] < td_avg.total_km.sum():
    st.write("»Consider running more for longer distances in the run up to the marathon")  
if year_user_df.loc[0,"total_km"] > td_avg.total_km.sum():
    st.write("»Consider running less for shorter distances overall in the run up to the marathon")

#
gsum = 0

for gs in range(len(nearest_neighbours_id)):
    gsum += out[nearest_neighbours_id[gs]].gap

gaverage = gsum/(gs+1)

gs_frame = {'gap': gaverage}
gs_avg = pd.DataFrame(gs_frame)

gs_avg['hashedathleteid'] = 'Other Runners'
gs_avg['weeks'] = np.arange(len(gs_avg))
graph5 = pd.concat([weekly_user_df, gs_avg], axis=0, ignore_index=True)

g5 = sns.lmplot(x="weeks", y="gap", hue="hashedathleteid",
                height=5, data=graph5, x_ci="sd")

# Use more informative axis labels than are provided by default
g5.set_axis_labels("Week", "Gap (Hours)")
st.subheader("Gap between runs in Hours")
st.write("This compares your average weekly gap between the runs (Hours) with your neighbour runners' average weekly gap between their runs. The higher the week number the closer to the marathon day.")
st.pyplot()

runnersMeanGap = round(gs_avg.gap.mean(),2)
yourMeanGap = round(year_user_df.loc[0, "gap"], 2)
st.write("Your average gap is: ", yourMeanGap, "Hours, with standard deviation: ", round(weekly_user_df.gap.std(), 2))
st.write("Other runners average gap is: ", runnersMeanGap, "Hours, with standard deviation: ", round(gs_avg.gap.std(),2))

if ((weekly_user_df.gap.std()- 2) > gs_avg.gap.std()) or ((weekly_user_df.gap.std()+2) < gs_avg.gap.std()):
    st.write("»Consider running more consistently with fixed gaps")
if yourMeanGap > runnersMeanGap:
    st.write("»Consider running more frequently for better perfomance")  
if yourMeanGap < runnersMeanGap:
    st.write("»Consider running less frquently to prevent physical fatigue")

#
runsum = 0
runsum20 = 0
runsum30 = 0

for rs in range(len(nearest_neighbours_id)):
    runsum += out[nearest_neighbours_id[rs]].num_runs
    runsum20 += out[nearest_neighbours_id[rs]].num_20k
    runsum30 += out[nearest_neighbours_id[rs]].num_30k

runaverage = runsum/(rs+1)
runaverage20 = runsum20/(rs+1)
runaverage30 = runsum30/(rs+1)

rs_frame = {'num_runs': runaverage}
rs_frame20 = {'num_20k': runaverage20}
rs_frame30 = {'num_30k': runaverage30}

rs_avg = pd.DataFrame(rs_frame)
rs_avg20 = pd.DataFrame(rs_frame20)
rs_avg30 = pd.DataFrame(rs_frame30)
rs_avg['num_20k'] = rs_avg20
rs_avg['num_30k'] = rs_avg30
rs_avg = rs_avg.round()
#rs_avg['hashedathleteid']='Other Runners'
#rs_avg['weeks'] = np.arange(len(rs_avg))
st.subheader("Other Runners Run Types")
st.write("Note: Week 13 is final week before marathon.")
st.write("Other runners run types:")
st.write(rs_avg)
rs_user = weekly_user_df[["num_runs", "num_20k", "num_30k"]]
st.write("Your Run Types:")
st.write(rs_user)
st.write("Difference:")
st.write(rs_avg - rs_user)
st.write("»It is recommended to do the longer runs around 12 weeks ahead of the marathon")


st.text("\n");st.text("\n");st.text("\n");st.text("\n");st.text("\n")

import random


st.title("COMPARE TO A RANDOM NEIGHBOUR RUNNER")
st.text("Note: RR stands for Random Runner")
#Picking a random runner
n = random.randint(0,len(nearest_neighbours_id)-1)


figz = plt.figure()

st.subheader("Pace Comparison")
for frame in [out[nearest_neighbours_id[n]]]:
    plt.plot(frame['average_pace'], 'r',label="RR Average")
    plt.plot(frame['max_pace'], 'y',label="RR Max")
    plt.plot(frame['min_pace'], 'm', label="RR Min")
    
for frame in [weekly_user_df]:
    plt.plot(frame['average_pace'], 'b',label="Your Average")
    plt.plot(frame['max_pace'], 'c',label="Your Max")
    plt.plot(frame['min_pace'], 'k', label="Your Min")
    

plt.xlabel("Weeks")
plt.ylabel("Pace")
#plt.legend("YYYNNN")
#leg = plt.legend((line1, line2, line3), ('label1', 'label2', 'label3'))
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('xx-small')
plt.legend(bbox_to_anchor=(0.9, 1), loc='upper left', prop=fontP)
st.pyplot()
st.write("YOU:")
st.write("Top speed:", round(max(weekly_user_df.max_pace),2),"mph, Minimum speed:", round(min(weekly_user_df.min_pace),2),"mph, Average Pace:", round(weekly_user_df.average_pace.mean(),2), "mph, with Standard deviation:", round(weekly_user_df.average_pace.std(),2))
st.write("RANDOM RUNNER:")
st.write("Top speed:", round(max(out[nearest_neighbours_id[n]].max_pace),2),"mph, Minimum speed:", round(min(out[nearest_neighbours_id[n]].min_pace),2),"mph, Average Pace:", round(out[nearest_neighbours_id[n]].average_pace.mean(),2), "mph, with Standard deviation", round(out[nearest_neighbours_id[n]].average_pace.std(),2))


st.subheader("Distance Comparison")
#####Distance
for frame in [out[nearest_neighbours_id[n]]]:
    plt.plot(frame['total_km'], 'k',label="RR Total")
    plt.plot(frame['average_km'], 'r',label="RR Average")
    plt.plot(frame['max_km'], 'y',label="RR Max")
    plt.plot(frame['min_km'], 'm', label="RR Min")
    
for frame in [weekly_user_df]:
    plt.plot(frame['total_km'], 'g',label="Your Total")
    plt.plot(frame['average_km'], 'b',label="Your Average")
    plt.plot(frame['max_km'], 'c',label="Your Max")
    plt.plot(frame['min_km'], 'k', label="Your Min")
    

plt.xlabel("Weeks")
plt.ylabel("Distance (KM)")

from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('xx-small')
plt.legend(bbox_to_anchor=(0.9, 1), loc='upper left', prop=fontP)
st.pyplot()
st.write("YOU:")
st.write("Longest Run:", round(max(weekly_user_df.max_km),2),"KM, Shortest run:", round(min(weekly_user_df.min_km),2),"KM, Average Distance in a run:", round(weekly_user_df.average_km.mean(),2), "KM, with Standard deviation:", round(weekly_user_df.average_km.std(),2))
st.write("RANDOM RUNNER:")
st.write("Longest Run:", round(max(out[nearest_neighbours_id[n]].max_km),2),"KM, Shortest run:", round(min(out[nearest_neighbours_id[n]].min_km),2),"KM, Average Distance in a run:", round(out[nearest_neighbours_id[n]].average_km.mean(),2), "KM, with Standard deviation", round(out[nearest_neighbours_id[n]].average_km.std(),2))


#####Gap
st.subheader("Gap Comparison")
for frame in [out[nearest_neighbours_id[n]]]:
    plt.plot(frame['gap'], 'r',label="RR Gap")
    
for frame in [weekly_user_df]:
    plt.plot(frame['gap'], 'b',label="Your Gap")
    

plt.xlabel("Weeks")
plt.ylabel("Gap Between Runs (Hours)")

from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('xx-small')
plt.legend(bbox_to_anchor=(0.9, 1), loc='upper left', prop=fontP)
st.pyplot()
st.write("YOU:")
st.write("Longest Gap:", round(max(weekly_user_df.gap),2),"Hours, Shortest gap:", round(min(weekly_user_df.gap),2),"Hours, Average gap between runs:", round(weekly_user_df.gap.mean(),2), "Hours, with Standard deviation:", round(weekly_user_df.gap.std(),2))
st.write("RANDOM RUNNER:")
st.write("Longest Gap:", round(max(out[nearest_neighbours_id[n]].gap),2),"Hours, Shortest gap:", round(min(out[nearest_neighbours_id[n]].gap),2),"Hours, Average gap between runs:", round(out[nearest_neighbours_id[n]].gap.mean(),2), "Hours, with Standard deviation", round(out[nearest_neighbours_id[n]].gap.std(),2))

st.subheader("Run Types Comparison")
st.write("YOU:")
st.write("Total Number of runs:", weekly_user_df.num_runs.sum(), " ; Runs 20-30KM:", weekly_user_df.num_20k.sum(), " ; Runs 30KM+:", weekly_user_df.num_30k.sum())
st.write("RANDOM RUNNER:")
st.write("Total Number of runs:", out[nearest_neighbours_id[n]].num_runs.sum(), " ; Runs 20-30KM:", out[nearest_neighbours_id[n]].num_20k.sum(), " ; Runs 30KM+:", out[nearest_neighbours_id[n]].num_30k.sum())

st.text("\n");st.text("\n");st.text("\n");st.text("\n");st.text("\n")

if st.button('SHUFFLE NEIGHBOUR'):
    st.title("COMPARE TO A RANDOM NEIGHBOUR RUNNER")
    st.text("Note: RR stands for Random Runner")
    #Picking a random runner
    n = random.randint(0,len(nearest_neighbours_id)-1)


    figz = plt.figure()

    st.subheader("Pace Comparison")
    for frame in [out[nearest_neighbours_id[n]]]:
        plt.plot(frame['average_pace'], 'r',label="RR Average")
        plt.plot(frame['max_pace'], 'y',label="RR Max")
        plt.plot(frame['min_pace'], 'm', label="RR Min")
    
    for frame in [weekly_user_df]:
        plt.plot(frame['average_pace'], 'b',label="Your Average")
        plt.plot(frame['max_pace'], 'c',label="Your Max")
        plt.plot(frame['min_pace'], 'k', label="Your Min")
    

    plt.xlabel("Weeks")
    plt.ylabel("Pace")
    
    from matplotlib.font_manager import FontProperties

    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(bbox_to_anchor=(0.9, 1), loc='upper left', prop=fontP)
    st.pyplot()
    st.write("YOU:")
    st.write("Top speed:", round(max(weekly_user_df.max_pace),2),"mph, Minimum speed:", round(min(weekly_user_df.min_pace),2),"mph, Average Pace:", round(weekly_user_df.average_pace.mean(),2), "mph, with Standard deviation:", round(weekly_user_df.average_pace.std(),2))
    st.write("RANDOM RUNNER:")
    st.write("Top speed:", round(max(out[nearest_neighbours_id[n]].max_pace),2),"mph, Minimum speed:", round(min(out[nearest_neighbours_id[n]].min_pace),2),"mph, Average Pace:", round(out[nearest_neighbours_id[n]].average_pace.mean(),2), "mph, with Standard deviation", round(out[nearest_neighbours_id[n]].average_pace.std(),2))


    st.subheader("Distance Comparison")
    #####Distance
    for frame in [out[nearest_neighbours_id[n]]]:
        plt.plot(frame['total_km'], 'k',label="RR Total")
        plt.plot(frame['average_km'], 'r',label="RR Average")
        plt.plot(frame['max_km'], 'y',label="RR Max")
        plt.plot(frame['min_km'], 'm', label="RR Min")
    
    for frame in [weekly_user_df]:
        plt.plot(frame['total_km'], 'g',label="Your Total")
        plt.plot(frame['average_km'], 'b',label="Your Average")
        plt.plot(frame['max_km'], 'c',label="Your Max")
        plt.plot(frame['min_km'], 'k', label="Your Min")
    

    plt.xlabel("Weeks")
    plt.ylabel("Distance (KM)")

    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(bbox_to_anchor=(0.9, 1), loc='upper left', prop=fontP)
    st.pyplot()
    st.write("YOU:")
    st.write("Longest Run:", round(max(weekly_user_df.max_km),2),"KM, Shortest run:", round(min(weekly_user_df.min_km),2),"KM, Average Distance in a run:", round(weekly_user_df.average_km.mean(),2), "KM, with Standard deviation:", round(weekly_user_df.average_km.std(),2))
    st.write("RANDOM RUNNER:")
    st.write("Longest Run:", round(max(out[nearest_neighbours_id[n]].max_km),2),"KM, Shortest run:", round(min(out[nearest_neighbours_id[n]].min_km),2),"KM, Average Distance in a run:", round(out[nearest_neighbours_id[n]].average_km.mean(),2), "KM, with Standard deviation", round(out[nearest_neighbours_id[n]].average_km.std(),2))


    #####Gap
    st.subheader("Gap Comparison")
    for frame in [out[nearest_neighbours_id[n]]]:
        plt.plot(frame['gap'], 'r',label="RR Gap")
    
    for frame in [weekly_user_df]:
        plt.plot(frame['gap'], 'b',label="Your Gap")
    

    plt.xlabel("Weeks")
    plt.ylabel("Gap Between Runs (Hours)")

    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(bbox_to_anchor=(0.9, 1), loc='upper left', prop=fontP)
    st.pyplot()
    st.write("YOU:")
    st.write("Longest Gap:", round(max(weekly_user_df.gap),2),"Hours, Shortest gap:", round(min(weekly_user_df.gap),2),"Hours, Average gap between runs:", round(weekly_user_df.gap.mean(),2), "Hours, with Standard deviation:", round(weekly_user_df.gap.std(),2))
    st.write("RANDOM RUNNER:")
    st.write("Longest Gap:", round(max(out[nearest_neighbours_id[n]].gap),2),"Hours, Shortest gap:", round(min(out[nearest_neighbours_id[n]].gap),2),"Hours, Average gap between runs:", round(out[nearest_neighbours_id[n]].gap.mean(),2), "Hours, with Standard deviation", round(out[nearest_neighbours_id[n]].gap.std(),2))

    st.subheader("Run Types Comparison")
    st.write("YOU:")
    st.write("Total Number of runs:", weekly_user_df.num_runs.sum(), " ; Runs 20-30KM:", weekly_user_df.num_20k.sum(), " ; Runs 30KM+:", weekly_user_df.num_30k.sum())
    st.write("RANDOM RUNNER:")
    st.write("Total Number of runs:", out[nearest_neighbours_id[n]].num_runs.sum(), " ; Runs 20-30KM:", out[nearest_neighbours_id[n]].num_20k.sum(), " ; Runs 30KM+:", out[nearest_neighbours_id[n]].num_30k.sum())

    st.text("\n");st.text("\n");st.text("\n");st.text("\n");st.text("\n")

else:
    st.write("If you would like to compare with another neighbour click the button above.")


st.text("\n");st.text("\n");st.text("\n");st.text("\n");st.text("\n")

st.subheader("Useful Links")
st.write("Starva: https://www.strava.com/")
st.write("Streamlit: https://www.streamlit.io/")
st.write("Marathons Ireland: https://worldsmarathons.com/c/marathon/ireland")
st.write("How to run a marathon: https://www.rei.com/learn/expert-advice/training-for-your-first-marathon.html")

st.text("\n");st.text("\n")
st.text("Developed by Ahmed Jouda\nSeptember 2020")  

