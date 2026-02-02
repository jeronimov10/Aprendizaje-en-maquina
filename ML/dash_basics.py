import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output


df = pd.read_csv(r'ML\ventas_temporales_dataset.csv')

""" 
fig = px.pie(df, values='precio_usd', names='temporada_alta', title='Grafica precio vs temporada alta')





app = dash.Dash(__name__)

app.layout = html.Div(children=[html.H1('Dashboard', style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
                                html.P('Proporcion precio vs temporada alta o baja', style={'textAlign':'center', 'color': '#F57241'}),
                                dcc.Graph(figure=fig),
                                               
                    ])
# Run the application                   
if __name__ == '__main__':
    app.run() 
    
"""

app = dash.Dash(__name__)

app.layout = html.Div(children=[ html.H1('Dashboard Ventas por Región',style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
                                html.Div(["Input Year: ", dcc.Input(id='input-year', value='2018',
                                type='number', style={'height':'50px', 'font-size': 35}),],
                                style={'font-size': 40}),
                                html.Br(),
                                html.Br(),
                                html.Div(dcc.Graph(id='line-plot')),
                                ])
# add callback decorator
@app.callback( Output(component_id='line-plot', component_property='figure'),
               Input(component_id='input-year', component_property='value'))

def get_graph(entered_year):

    df_a = df[df['anio'] == int(entered_year)]

    # Agrupar por región y sumar ingresos
    ingresos_region = df_a.groupby('region')['ingresos_usd'].sum().reset_index()

    fig = go.Figure(data=go.Bar(x=ingresos_region['region'], y=ingresos_region['ingresos_usd'], marker=dict(color='green')))
    fig.update_layout(title=f'Ingresos USD por Región - {entered_year}', xaxis_title='Región', yaxis_title='Ingresos USD')
    return fig
# Run the app
if __name__ == '__main__':
    app.run()
