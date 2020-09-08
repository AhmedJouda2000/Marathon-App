import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import random

os.chdir("C:\\Users\\ahmed\\Desktop")

st.title("RUNNING TRAINING APP")
title = st.text_input('Select Your Country', 'is')
st.write('The current country selected is', title)
currentDistance = st.slider('How long do you currently run for?', 5, 60, 10)
st.write('Distance range considered: ', currentDistance-2, '-', currentDistance+2, 'KM')
projectedDistance = st.slider('How many KMs is your aim?', 5, 60, 20)
df_original = pd.read_pickle('df_merged_ie.pkl')
df_original = df_original[df_original.totaldistance < 50000]
df_original = df_original[df_original.totaldistance > 100]

df = df_original[df_original.totaldistance < 1000*(currentDistance+2)]
df = df[df.totaldistance > 1000*(currentDistance-2)]
df = df[df.country_code == title]
if st.checkbox('Show Raw Data'):
    'Raw Data', df

#df.rename(columns = {'startlatapprox': 'lat'}, inplace = True)
#df.rename(columns = {'startlngapprox': 'lon'}, inplace = True)

df2 = df.drop(columns= 'city_name')
df2.rename(columns = {'startlatapprox': 'lat'}, inplace = True)
df2.rename(columns = {'startlngapprox': 'lon'}, inplace = True)
st.write('Where people ran around', currentDistance, 'KM')
st.map(df2)
df2.sort_values(by=['hashedathleteid'], inplace=True)

df3 = df_original[df_original.totaldistance < 1000*(projectedDistance+2)]
df3 = df3[df3.totaldistance > 1000*(projectedDistance-2)]
df3 = df3[df3.country_code == 'is']
df3 = df3.drop(columns= 'city_name')
df3.rename(columns = {'startlatapprox': 'lat'}, inplace = True)
df3.rename(columns = {'startlngapprox': 'lon'}, inplace = True)
st.write('Where people ran around', projectedDistance, 'KM')
st.map(df3)
df3.sort_values(by=['hashedathleteid'], inplace=True)

s2 = pd.merge(df2, df3, on='hashedathleteid')
s2 = s2.drop_duplicates(subset='hashedathleteid', keep='first')

list_of_ids = s2['hashedathleteid'].to_list()
st.write(list_of_ids)
miniOriginal = df_original[['hashedathleteid','startdatelocal','totaldistance']]
#requiredid = '0d40fd9180b8ab80782b2604ba24ab0a0e377872f3c8f5a052e44ba5cf758f3b'
#for x in range(0, 2):
final = miniOriginal[miniOriginal['hashedathleteid'].str.contains(list_of_ids[0])]
finala = final[['totaldistance']]
#final_1 = final_1.set_index('startdatelocal')
finala = finala.reset_index()
finala = finala.drop(columns= 'index')
finala.rename(columns = {'totaldistance': 'Runner A'}, inplace = True)

finalb = miniOriginal[miniOriginal['hashedathleteid'].str.contains(list_of_ids[2])]
finalb = finalb[['totaldistance']]
finalb = finalb.reset_index()
finalb = finalb.drop(columns= 'index')
finalb.rename(columns = {'totaldistance': 'Runner B'}, inplace = True)
st.write(finalb.count())
 
finalc = miniOriginal[miniOriginal['hashedathleteid'].str.contains(list_of_ids[3])]
finalc = finalc[['totaldistance']]
finalc = finalc.reset_index()
finalc = finalc.drop(columns= 'index')
finalc.rename(columns = {'totaldistance': 'Runner C'}, inplace = True)
st.write(finalc.count())

finald = miniOriginal[miniOriginal['hashedathleteid'].str.contains(list_of_ids[4])]
finald = finald[['totaldistance']]
finald = finald.reset_index()
finald = finald.drop(columns= 'index')
finald.rename(columns = {'totaldistance': 'Runner D'}, inplace = True)
st.write(finald.count())

finale = miniOriginal[miniOriginal['hashedathleteid'].str.contains(list_of_ids[5])]
finale = finale[['totaldistance']]
finale = finale.reset_index()
finale = finale.drop(columns= 'index')
finale.rename(columns = {'totaldistance': 'Runner E'}, inplace = True)
st.write(finale.count())
 
from functools import reduce
finals = [finala, finalb, finalc, finald, finale]
lastofall = reduce(lambda left,right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), finals)
#lastofall = pd.merge(finala, finalb, how='outer', left_index=True, right_index=True)
#final_1 = final_1.join(tmp)
st.line_chart(lastofall)
st.write(lastofall.count())
st.write(lastofall.head(10))

#averages
st.write('Mean Distance for Runners in KM')
av_column = lastofall.mean(axis=0)
st.write(av_column)

st.write('Average practice trend')
av_row = lastofall.mean(axis=1)
st.line_chart(av_row)