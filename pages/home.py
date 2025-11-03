from dash import html
import dash_bootstrap_components as dbc

layout = dbc.Container([
    html.H1(
        "Bem-vindo ao Dashboard de Análise de Risco de Fogo - Cerrado",
        className="text-center my-4"
    ),

    html.P(
        "Nesta página, explicaremos a estrutura de dados utilizada para analisar casos de incêndio e seu risco de fogo. "
        "Use os botões abaixo para acessar as diferentes seções do sistema.",
        className="text-center mb-5"
    ),

    # Seção dos botões principais
    html.Div([
        dbc.Button("📊 Gráficos por Ano", color="primary", href="/graficos", className="mx-2 my-1", size="lg"),
        dbc.Button("🧮 Modelo de Previsão", color="success", href="/dados", className="mx-2 my-1", size="lg"),
        dbc.Button("📈 Previsão 2026", color="warning", href="/mapa", className="mx-2 my-1", size="lg"),
    ], className="d-flex justify-content-center flex-wrap"),

], className="mt-5")
