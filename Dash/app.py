import os
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px

app = dash.Dash()
os.chdir("C:\\Users\\ahmed\\Desktop")
df = pd.read_pickle('df_merged_ie.pkl')
os.chdir("C:\\Users\\ahmed\\Desktop\\Dash")
df = df[df.totaldistance > 1000]
df = df[df.totaldistance < 10000]
df = df[df.country_code == 'is']
#fig = px.scatter(df, x="startdatelocal", y="totaldistance")

app.layout = html.Div[(
        dcc.Graph(
                id='mapbox',
                figure={
                    'df': [go.Bar(
                            x = df ['startdatelocal'],
                            y = df ['totaldistance'],
                            )
                    ],
                    'layout': go.Layout(
                        legend={'x': 0, 'y': 1},
                        hovermode= 'closest'
                    )
                }
        )
    )]
#app.layout = html.Div[(
#    dcc.Graph(
#        id = 'Scatter_Chart',
#        figure = {
#            'data' : [
#                go.Scatter(
#                    x = df.totaldistance,
#                    y = df.startdatelocal,
#                    mode = 'markers'
#                )
#            ],
#            'layout' : go.Layout(
#                title = 'Scatterplot of distance vs time',
#                xaxis = {'title' : 'Distance'},
#                yaxis = {'title' : 'Time'}
#            )
#        }
#    )
#)]


if __name__ == '__main__':
    app.run_server(debug = True)