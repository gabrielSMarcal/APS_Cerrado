from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
from app import app
from models.GraficoInterativo import GraficoInterativo
from render.grafico_media_risco import gerar_grafico_media_risco_por_ano
import plotly.graph_objects as go

# Inicializar a classe de gráficos interativos
graficos = GraficoInterativo()

# Obter lista de graficos (compatível com estrutura anterior)
lista_graficos = graficos.obter_lista_graficos()

# Definir valor padrão (ano mais recente)
valor_padrao = graficos.ano_mais_recente

layout = dbc.Container([
    dcc.Store(id='ano-selecionado-store', data=valor_padrao),
    html.H3("graficos por Ano", className="text-center my-4"),
    dbc.Row([
        # Coluna lateral com lista de anos
        dbc.Col([
            html.Div([
                html.H5("Selecione o Ano", className="text-center mb-3"),
                html.Div([
                    dbc.Button(
                        f"Ano {grafico['ano']}",
                        id={'type': 'btn-ano', 'index': grafico['ano']},
                        color="primary" if grafico['ano'] == valor_padrao else "secondary",
                        outline=grafico['ano'] != valor_padrao,
                        className="w-100 mb-2",
                        size="lg",
                        n_clicks=0
                    ) for grafico in lista_graficos
                ]),
            ], className="sticky-top", style={'top': '20px'})
        ], width=2, className="pe-3"),
        
        # Coluna principal com os gráficos
        dbc.Col([
            # Gráfico de pontos por município
            html.Div([
                html.H5("Risco de Fogo Anual", className="text-center mb-3"),
                dcc.Graph(id='grafico-ano', style={'height': '70vh'})
            ], className="mb-4"),
            
            # Gráfico de média de risco por estado
            html.Div([
                html.H5("Média de Risco por Estado", className="text-center mb-3"),
                dcc.Graph(id='grafico-media-risco', style={'height': '70vh'})
            ])
        ], width=10)
    ])
], fluid=True, className="px-4")

@app.callback(
    Output('ano-selecionado-store', 'data'),
    Input({'type': 'btn-ano', 'index': ALL}, 'n_clicks'),
    State('ano-selecionado-store', 'data'),
    prevent_initial_call=True
)
def update_ano_selecionado(n_clicks, ano_atual):
    '''
    Atualiza o ano selecionado quando um botão é clicado.
    
    Parâmetros:
        n_clicks: Lista de cliques dos botões
        ano_atual: Ano atualmente selecionado
        
    Retorno:
        Ano selecionado atualizado
    '''
    if ctx.triggered_id:
        return ctx.triggered_id['index']
    
    return ano_atual

@app.callback(
    [Output('grafico-ano', 'figure'),
     Output('grafico-media-risco', 'figure'),
     Output({'type': 'btn-ano', 'index': ALL}, 'color'),
     Output({'type': 'btn-ano', 'index': ALL}, 'outline')],
    Input('ano-selecionado-store', 'data'),
    prevent_initial_call=False
)
def update_grafico_display(ano_selecionado):
    '''
    Atualiza a exibição dos gráficos e estilos dos botões.
    '''
    
    # Obter figuras do cache (agilizar carregamento)
    figura_pontos = graficos.obter_figura(ano_selecionado)
    figura_media = gerar_grafico_media_risco_por_ano(ano_selecionado)
    
    # Se não encontrar, usar o ano mais recente
    if figura_pontos is None:
        ano_selecionado = graficos.ano_mais_recente
        figura_pontos = graficos.obter_figura(ano_selecionado)
        figura_media = gerar_grafico_media_risco_por_ano(ano_selecionado)
    
    # Se ainda não houver figura de média, criar uma figura vazia
    if figura_media is None:
        figura_media = go.Figure()
        figura_media.update_layout(
            title=f"Sem dados disponíveis para o ano {ano_selecionado}",
            annotations=[{
                'text': 'Dados não disponíveis',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 20}
            }]
        )
    
    # Atualizar cores e outlines dos botões
    colors = ['primary' if grafico['ano'] == ano_selecionado else 'secondary' for grafico in lista_graficos]
    outlines = [False if grafico['ano'] == ano_selecionado else True for grafico in lista_graficos]
    
    return figura_pontos, figura_media, colors, outlines