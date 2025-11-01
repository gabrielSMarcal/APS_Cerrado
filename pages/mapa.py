from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

# Carregando o CSV de previsão
df_previsao = pd.read_csv('data/treated_db/previsao_2026.csv')
df_previsao['Data'] = pd.to_datetime(df_previsao['Data'])

# Criar coluna apenas com a data formatada (sem hora)
df_previsao['DataHora'] = df_previsao['Data'].dt.date

# Criar o mapa
fig = px.scatter_map(
    df_previsao,
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
    zoom=4,
    title="Previsão de Risco de Fogo no Cerrado - 2026"
)

# Layout da página
layout = dbc.Container([
    html.H3("Previsão de Queimadas no Cerrado - 2026", className="text-center my-4"),
    dcc.Graph(figure=fig, style={'height': '80vh'})
], fluid=True)

# 