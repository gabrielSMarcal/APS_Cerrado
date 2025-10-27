from dash import html
import dash_bootstrap_components as dbc

layout = dbc.Container([
    html.H1("Bem-vindo ao Dashboard de Análise de Risco de Fogo - Cerrado", className="text-center my-4"),
    html.P("Esta é a página inicial. Use a barra de navegação acima para acessar o Mapa Interativo e os Grafos por Ano.", className="text-center"),
    dbc.Alert("Estrutura de página reestruturada com sucesso!", color="success", className="mt-4")
], className="mt-5")
