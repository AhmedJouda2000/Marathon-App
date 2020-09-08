import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output

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

app.layout = html.Div(
    [
    
    html.Div([
         html.Label('This is an input box'),
    dcc.Input(
        id = 'countryinput',
        placeholder = 'Input country',
        type = 'text',
        value = ''
    )
    ]),

    dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'country': 'is', 'value': 'is'},
            {'country': 'il', 'value': 'il'}
        ],
        value='is'
    ),

    html.H1(children='Runners in Iceland'),
    html.Div(children='''1k - 10k'''),
    dcc.Graph(id='example-graph', style={"width": "75%", "display": "inline-block"},
        figure=px.scatter(x=df["startdatelocal"], y=df["totaldistance"])),
    html.Div(id='my-output'),
    
   
    ]
)

@app.callback(
    Output(component_id='my-output', component_property='children'),
    [Input(component_id='countryinput', component_property='value')])

def my_callback_func(dropdown_value):
    
    os.chdir("C:\\Users\\ahmed\\Desktop")
    df = pd.read_pickle('df_merged_ie.pkl')
    os.chdir("C:\\Users\\ahmed\\Desktop\\Dash")
    df = df[df.totaldistance > 1000]
    df = df[df.totaldistance < 10000]
    #df = df[df.country_code == 'is']
    df = df[df['country_code'].eq(dropdown_value)]
    dcc.Graph(id='example-graph', style={"width": "75%", "display": "inline-block"},
        figure=px.scatter(x=df["startdatelocal"], y=df["totaldistance"])),

def update_output_div(input_value):
    return 'Output: {}'.format(input_value)

if __name__ == '__main__':
    app.run_server(debug=True)