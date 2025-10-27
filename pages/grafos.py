from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from app import app
from lista_grafos import gerar_lista_grafos

# Gerar lista de grafos por ano
lista_grafos = gerar_lista_grafos()

# Criar opções para o dropdown
opcoes_anos = [{'label': f'Ano {grafo["ano"]}', 'value': grafo["ano"]} for grafo in lista_grafos]

# Definir valor padrão (2025)
valor_padrao = 2025 if any(g['ano'] == 2025 for g in lista_grafos) else lista_grafos[-1]['ano']

layout = dbc.Container([
    html.H3("Grafos por Ano", className="text-center my-4"),
    html.Div([
        html.H4("Selecione o Ano"),
        dcc.Dropdown(
            id='ano-dropdown',
            options=opcoes_anos,
            value=valor_padrao,
            clearable=False,
            style={'width': '50%', 'margin': 'auto'}
        ),
    ], className="mb-4 text-center"),
    html.Br(),
    dcc.Graph(id='grafo-ano')
], fluid=True)

# Callback para atualizar o grafo (o mesmo que estava no seu app.py)
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
