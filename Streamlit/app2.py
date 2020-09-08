#import matplotlib.pylab as plt
import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

os.chdir("C:\\Users\\ahmed\\Desktop")

st.title("RUNNING TRAINING APP")
title = st.text_input('Select Your Country', 'is')
st.write('The current country selected is', title)
currentDistance = st.slider('How long do you currently run for?', 5, 60, 10)
st.write('Distance range considered: ', currentDistance-2, '-', currentDistance+2, 'KM')
projectedDistance = st.slider('How many KMs is your aim?', 5, 60, 20)
df_original = pd.read_pickle('df_merged_ie.pkl')
#df = df.head(2)

df = df_original[df_original.totaldistance < 1000*(currentDistance+2)]
df = df[df.totaldistance > 1000*(currentDistance-2)]
df = df[df.country_code == title]
if st.checkbox('Show Raw Data'):
    'Raw Data', df

df.rename(columns = {'startlatapprox': 'lat'}, inplace = True)
df.rename(columns = {'startlngapprox': 'lon'}, inplace = True)

st.map(df)

#index = df.index
#numberrows = len(index)
st.write(df.count())

df1 = df[['startdatelocal','totaldistance']]
df1 = df1.set_index('startdatelocal')
st.write('People who run similar distances')
st.line_chart(df1)

st.write('Runners progression from your distance')
#here I must take some of the runners and show a map of their totaldistance vs time
#st.line_chart(df)

#activity_counts = df.groupby('hashedathleteid').agg('count')['totaldistance']
#st.line_chart(activity_counts)

