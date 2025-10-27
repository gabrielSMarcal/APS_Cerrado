from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from app import app # Importar a instância do Dash
from data.connection import connection # Importar a função de conexão de dados

# DataFrame principal
df = connection()

layout = dbc.Container([
    html.H3("Mapa Interativo", className="text-center my-4"),
    html.Div([
        html.H4("Filtrar por Data"),
        dcc.DatePickerRange(
            id='date-range-slider',
            min_date_allowed=df['Data'].min(),
            max_date_allowed=df['Data'].max(),
            start_date=df['Data'].min(),
            end_date=df['Data'].max(),
        ),
    ], className="mb-4"),
    dcc.Graph(id='forest_burn_map')
], fluid=True)

# Callback para atualizar o mapa (o mesmo que estava no seu app.py)
@app.callback(
    Output('forest_burn_map', 'figure'),
    Input('date-range-slider', 'start_date'),
    Input('date-range-slider', 'end_date')
)
def update_map(start_date, end_date):
    # ... (seu código original do callback update_map)
    # Certifique-se de incluir todas as importações necessárias (como `plotly.express as px`)
    # ...
    if not start_date or not end_date:
        return {} 

    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()

    filtered_df = df[(df['Data'] >= start) & (df['Data'] <= end)]

    fig = px.scatter_map(
        filtered_df,
        lat="Latitude",      
        lon="Longitude",    
        color="RiscoFogo",
        color_continuous_scale=px.colors.sequential.Turbo,
        hover_name="Estado",
        hover_data={
            "Municipio": True,
            "DataHora": True,
            "DiaSemChuva": True,
            "Precipitacao": True,
            "FRP": True,
            "Latitude": False,
            "Longitude": False
        },
        map_style="carto-positron",
        zoom=4
    )
    
    return fig
