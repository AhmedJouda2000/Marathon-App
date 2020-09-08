
#import matplotlib.pylab as plt
import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

os.chdir("C:\\Users\\ahmed\\Desktop")

st.title("RUNNING TRAINING APP")
title = st.text_input('Select Country', 'is')
st.write('The current country selected is', title)
maxDistance = st.slider('Distance Range (Max)', 10, 60, 10)
st.write('Distance range shown: ', maxDistance-5, '-', maxDistance, 'KM')
df = pd.read_pickle('df_merged_ie.pkl')
#df = df.head(2)

df = df[df.totaldistance < 1000*maxDistance]
df = df[df.totaldistance > 1000*(maxDistance-5)]
df = df[df.country_code == title]
if st.checkbox('Show Raw Data'):
    'Raw Data', df

df.rename(columns = {'startlatapprox': 'lat'}, inplace = True)
df.rename(columns = {'startlngapprox': 'lon'}, inplace = True)

st.map(df)

#index = df.index
#numberrows = len(index)
st.write(df.count())

