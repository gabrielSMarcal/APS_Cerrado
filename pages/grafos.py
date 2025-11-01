from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
from app import app
from lista_grafos import gerar_lista_grafos

# Gerar lista de grafos por ano
lista_grafos = gerar_lista_grafos()

# Definir valor padrão (2025)
valor_padrao = 2025 if any(g['ano'] == 2025 for g in lista_grafos) else lista_grafos[-1]['ano']

layout = dbc.Container([
    dcc.Store(id='ano-selecionado-store', data=valor_padrao),
    html.H3("Grafos por Ano", className="text-center my-4"),
    dbc.Row([
        # Coluna lateral com lista de anos
        dbc.Col([
            html.Div([
                html.H5("Selecione o Ano", className="text-center mb-3"),
                html.Div([
                    dbc.Button(
                        f"Ano {grafo['ano']}",
                        id={'type': 'btn-ano', 'index': grafo['ano']},
                        color="primary" if grafo['ano'] == valor_padrao else "secondary",
                        outline=grafo['ano'] != valor_padrao,
                        className="w-100 mb-2",
                        size="lg",
                        n_clicks=0
                    ) for grafo in lista_grafos
                ]),
            ], className="sticky-top", style={'top': '20px'})
        ], width=2, className="pe-3"),
        
        # Coluna principal com o gráfico
        dbc.Col([
            dcc.Graph(id='grafo-ano', style={'height': '80vh'})
        ], width=10)
    ])
], fluid=True, className="px-4")

# Callback para atualizar o store com o ano selecionado
@app.callback(
    Output('ano-selecionado-store', 'data'),
    Input({'type': 'btn-ano', 'index': ALL}, 'n_clicks'),
    State('ano-selecionado-store', 'data'),
    prevent_initial_call=True
)
def update_ano_selecionado(n_clicks, ano_atual):
    """
    Atualiza o ano selecionado quando um botão é clicado
    """
    from dash import ctx
    
    if ctx.triggered_id:
        return ctx.triggered_id['index']
    return ano_atual

# Callback para atualizar o grafo e os estilos dos botões
@app.callback(
    [Output('grafo-ano', 'figure'),
     Output({'type': 'btn-ano', 'index': ALL}, 'color'),
     Output({'type': 'btn-ano', 'index': ALL}, 'outline')],
    Input('ano-selecionado-store', 'data'),
    prevent_initial_call=False
)
def update_grafo_display(ano_selecionado):
    """
    Atualiza a exibição do gráfico e estilos dos botões
    """
    # Encontrar o gráfico correspondente
    figura = None
    for grafo in lista_grafos:
        if grafo['ano'] == ano_selecionado:
            figura = grafo['figura']
            break
    
    # Se não encontrar, usar o último da lista
    if figura is None:
        ano_selecionado = lista_grafos[-1]['ano'] if lista_grafos else valor_padrao
        figura = lista_grafos[-1]['figura'] if lista_grafos else {}
    
    # Atualizar cores e outlines dos botões
    colors = ['primary' if grafo['ano'] == ano_selecionado else 'secondary' for grafo in lista_grafos]
    outlines = [False if grafo['ano'] == ano_selecionado else True for grafo in lista_grafos]
    
    return figura, colors, outlines
