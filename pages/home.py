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
    
    html.P(
        "Os dados analisados incluem informações geográficas, temporais e ambientais relacionadas a incêndios no bioma Cerrado. "
        "A análise é realizada utilizando um modelo de aprendizado de máquina que incorpora um TAD (Tipo Abstrato de Dados) em forma de grafo "
        "para capturar relações espaciais e temporais entre os dados.",
        className="text-center mb-4"
    ),
    
    html.P(
        "Selecione uma opção para navegar pelo sistema:",
        className="text-center mb-4"
    ),
    
    html.Ul(
        [
            html.Li("📊 Gráficos por Ano: Visualize gráficos interativos que mostram a média de risco de fogo e a contagem de casos por estado no Cerrado para cada ano disponível."),
            html.Li("🧮 Modelo de Previsão: Acesse informações detalhadas sobre o modelo de aprendizado de máquina utilizado para prever o risco de fogo, incluindo métricas de desempenho e importância das features."),
            html.Li("📈 Previsão 2026: Veja a previsão do risco de fogo para o ano de 2026, gerada pelo modelo treinado com dados históricos e utilizando o TAD em forma de grafo."),
        ],
        className="mb-5"
    ),

    # Seção dos botões principais
    html.Div([
        dbc.Button("📊 Gráficos por Ano", color="primary", href="/graficos", className="mx-2 my-1", size="lg"),
        dbc.Button("🧮 Modelo de Previsão", color="success", href="/dados", className="mx-2 my-1", size="lg"),
        dbc.Button("📈 Previsão 2026", color="warning", href="/mapa", className="mx-2 my-1", size="lg"),
    ], className="d-flex justify-content-center flex-wrap"),

], className="mt-5")
