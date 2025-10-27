import plotly.express as px
import pandas as pd
import dash
from dash import Dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from data.connection import connection
from lista_grafos import gerar_lista_grafos

# DataFrame principal
df = connection()

# Gerar lista de grafos por ano
lista_grafos = gerar_lista_grafos()

# Criar opções para o dropdown
opcoes_anos = [{'label': f'Ano {grafo["ano"]}', 'value': grafo["ano"]} for grafo in lista_grafos]

# Definir valor padrão (2025)
valor_padrao = 2025 if any(g['ano'] == 2025 for g in lista_grafos) else lista_grafos[-1]['ano']

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"
    ],
    suppress_callback_exceptions=True
)

app.layout = html.Div([
    html.H1("Análise de Risco de Fogo - Cerrado", style={'textAlign': 'center'}),
    
    # Tabs para alternar entre mapa interativo e grafos por ano
    dcc.Tabs(id='tabs', value='tab-interativo', children=[
        dcc.Tab(label='Mapa Interativo', value='tab-interativo'),
        dcc.Tab(label='Grafos por Ano', value='tab-grafos'),
    ]),
    
    html.Div(id='tabs-content')
])

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-interativo':
        return html.Div([
            html.H3("Filtrar por Data"),
            dcc.DatePickerRange(
                id='date-range-slider',
                min_date_allowed=df['Data'].min(),
                max_date_allowed=df['Data'].max(),
                start_date=df['Data'].min(),
                end_date=df['Data'].max(),
            ),
            dcc.Graph(id='forest_burn_map')
        ])
    
    elif tab == 'tab-grafos':
        return html.Div([
            html.H3("Selecione o Ano"),
            dcc.Dropdown(
                id='ano-dropdown',
                options=opcoes_anos,
                value=valor_padrao,  # 2025 como padrão
                clearable=False,
                style={'width': '50%', 'margin': 'auto'}
            ),
            html.Br(),
            dcc.Graph(id='grafo-ano')
        ])

@app.callback(
    Output('forest_burn_map', 'figure'),
    Input('date-range-slider', 'start_date'),
    Input('date-range-slider', 'end_date')
)
def update_map(start_date, end_date):
    if not start_date or not end_date:
        return dash.no_update

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

@app.callback(
    Output('grafo-ano', 'figure'),
    Input('ano-dropdown', 'value')
)
def update_grafo_ano(ano_selecionado):
    """
    Retorna o gráfico correspondente ao ano selecionado
    """
    for grafo in lista_grafos:
        if grafo['ano'] == ano_selecionado:
            return grafo['figura']
    
    # Se não encontrar, retorna o último da lista
    return lista_grafos[-1]['figura'] if lista_grafos else {}

if __name__ == '__main__':
    app.run(debug=True)
