import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px

os.chdir("C:\\Users\\ahmed\\Desktop")
df = pd.read_pickle('df_merged_ie.pkl')
os.chdir("C:\\Users\\ahmed\\Desktop\\Dash")
df = df[df.totaldistance > 1000]
df = df[df.totaldistance < 10000]
df = df[df.country_code == 'is']
#pv = pd.pivot_table(df, index=['hashedathleteid'], columns=["Status"], values=['Quantity'], aggfunc=sum, fill_value=0)

#trace1 = go.Bar(x=pv.index, y=pv[('Quantity', 'declined')], name='Declined')
#trace2 = go.Bar(x=pv.index, y=pv[('Quantity', 'pending')], name='Pending')
#trace3 = go.Bar(x=pv.index, y=pv[('Quantity', 'presented')], name='Presented')
#trace4 = go.Bar(x=pv.index, y=pv[('Quantity', 'won')], name='Won')

app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children='Sales Funnel Report'),
    html.Div(children='''National Sales Funnel Report.'''),
    dcc.Graph(
        id='example-graph',
        figure={
            "data": [
                {
                    "x": df["startdatelocal"],
                    "y": df["totaldistance"],
                    "name" : "Total Distance",
                    "type": "line",
                    "marker": {"color": "#00ff00"},
                }                      
            ],

            "layout": {
                "showlegend": True,

                "xaxis": {
                    "automargin": True,
                    "title": {"text": "Date"}
                },
                "yaxis": {
                    "automargin": True,
                    "title": {"text": "Number"}
                    },
                "height": 550,
                "margin": {"t": 10, "l": 10, "r": 10},
            },
        },
    )
])

fig = px.scatter(x=df["startdatelocal"], y=df["totaldistance"])
fig.show()

if __name__ == '__main__':
    app.run_server(debug=True)