from dash import dcc, html, Output, Input, State
import dash_bootstrap_components as dbc

from app import app
from models.mapa_interativo import MapaInterativo

# Carregando o CSV de previsão
mapa = MapaInterativo('data/treated_db/previsao_2026.csv')

estados_df = mapa.carregar_estados()

# Criar coluna apenas com a data formatada (sem hora)
estados_df['DataHora'] = estados_df['Data'].dt.date

# Criar o mapa
layout = dbc.Container([
    # Título da página
    html.H3("Previsão de Queimadas no Cerrado - 2026", className="text-center my-4"),
    
    # Container principal com layout de duas colunas
    dbc.Row([
        # Coluna lateral - Filtros
        dbc.Col([
            html.Div([
                # Título da seção de filtros
                html.H5("Filtros", className="text-center mb-3"),
                
                # Botão para visualização geral (todos os estados)
                dbc.Button(
                    "🌎 Visualização Geral",
                    id="btn-todos-estados",
                    color="primary",
                    className="w-100 mb-3",
                    size="lg"
                ),
                
                html.Hr(),
                
                # Título da lista de estados
                html.H6("Selecione um Estado:", className="text-center mb-2"),
                
                # Lista de botões para cada estado
                html.Div([
                    dbc.Button(
                        estado,
                        id={'type': 'btn-estado', 'index': estado},
                        color="secondary",
                        outline=True,
                        className="w-100 mb-2",
                        size="sm"
                    ) for estado in estados_df
                ]),
                
                html.Hr(),
                
                # Painel de estatísticas
                html.H6("Estatísticas:", className="text-center mb-2"),
                html.Div(id='painel-estatisticas', className="p-3 bg-light rounded")
                
            ], className="sticky-top", style={'top': '20px'})
        ], width=3, className="pe-3"),
        
        # Coluna principal - Mapa
        dbc.Col([
            dcc.Graph(
                id='mapa-queimadas',
                figure=mapa.obter_figura(),
                style={'height': '85vh'}
            )
        ], width=9)
    ])
], fluid=True, className="px-4")


# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output('mapa-queimadas', 'figure'),
     Output('painel-estatisticas', 'children'),
     Output('btn-todos-estados', 'color'),
     Output('btn-todos-estados', 'outline'),
     Output({'type': 'btn-estado', 'index': dbc.ALL}, 'color'),
     Output({'type': 'btn-estado', 'index': dbc.ALL}, 'outline')],
    [Input('btn-todos-estados', 'n_clicks'),
     Input({'type': 'btn-estado', 'index': dbc.ALL}, 'n_clicks')],
    prevent_initial_call=False
)
def atualizar_mapa(n_clicks_geral, n_clicks_estados):
    """
    Callback principal para atualizar o mapa e estatísticas baseado no filtro selecionado.
    
    Args:
        n_clicks_geral: Número de cliques no botão "Visualização Geral"
        n_clicks_estados: Lista de cliques nos botões de estados
    
    Returns:
        Tupla com (figura, estatísticas, cores e outlines dos botões)
    """
    from dash import ctx
    
    # Determinar qual botão foi clicado
    triggered_id = ctx.triggered_id
    
    if triggered_id == 'btn-todos-estados':
        # Botão "Visualização Geral" foi clicado
        mapa.resetar_filtros()
        btn_geral_color = 'primary'
        btn_geral_outline = False
        btn_estados_colors = ['secondary'] * len(estados_df)
        btn_estados_outlines = [True] * len(estados_df)
        
    elif triggered_id and isinstance(triggered_id, dict) and triggered_id.get('type') == 'btn-estado':
        # Um botão de estado específico foi clicado
        estado_selecionado = triggered_id['index']
        mapa.filtrar_por_estado(estado_selecionado)
        
        btn_geral_color = 'secondary'
        btn_geral_outline = True
        
        # Atualizar cores dos botões de estado
        btn_estados_colors = []
        btn_estados_outlines = []
        for estado in estados_df:
            if estado == estado_selecionado:
                btn_estados_colors.append('primary')
                btn_estados_outlines.append(False)
            else:
                btn_estados_colors.append('secondary')
                btn_estados_outlines.append(True)
    else:
        # Carregamento inicial - sem filtro
        btn_geral_color = 'primary'
        btn_geral_outline = False
        btn_estados_colors = ['secondary'] * len(estados_df)
        btn_estados_outlines = [True] * len(estados_df)
    
    # Obter figura atualizada
    figura = mapa.obter_figura(force_refresh=True)
    
    # Obter estatísticas atualizadas
    stats = mapa.obter_estatisticas()
    
    # Criar painel de estatísticas com classificação de risco
    painel = html.Div([
        # Informações gerais
        html.P([
            html.Strong("Estado: "),
            stats['estado_filtrado'] or "Todos"
        ], className="mb-1"),
        html.P([
            html.Strong("Registros: "),
            f"{stats['total_registros']:,}"
        ], className="mb-1"),
        html.P([
            html.Strong("Municípios: "),
            f"{stats['municipios_unicos']}"
        ], className="mb-2"),
        
        html.Hr(className="my-2"),
        
        # Classificação de Risco
        html.H6("📈 Classificação de Risco", className="text-center mb-2"),
        
        # Risco Baixo (0-30) - Possível incêndio criminoso
        html.Div([
            html.P([
                html.Span("🟢 ", style={'fontSize': '1.2em'}),
                html.Strong("Baixo (0-30)")
            ], className="mb-0", style={'color': '#28a745'}),
            html.P(
                stats['risco_baixo']['descricao'],
                className="mb-0",
                style={'fontSize': '0.75em', 'fontStyle': 'italic', 'marginLeft': '1.5em'}
            ),
            html.P([
                f"{stats['risco_baixo']['quantidade']:,} registros ",
                html.Span(
                    f"({stats['risco_baixo']['percentual']:.1f}%)",
                    style={'fontWeight': 'bold'}
                )
            ], className="mb-2", style={'marginLeft': '1.5em'})
        ]),
        
        # Risco Médio (31-70)
        html.Div([
            html.P([
                html.Span("🟡 ", style={'fontSize': '1.2em'}),
                html.Strong("Médio (31-70)")
            ], className="mb-0", style={'color': '#ffc107'}),
            html.P(
                stats['risco_medio']['descricao'],
                className="mb-0",
                style={'fontSize': '0.75em', 'fontStyle': 'italic', 'marginLeft': '1.5em'}
            ),
            html.P([
                f"{stats['risco_medio']['quantidade']:,} registros ",
                html.Span(
                    f"({stats['risco_medio']['percentual']:.1f}%)",
                    style={'fontWeight': 'bold'}
                )
            ], className="mb-2", style={'marginLeft': '1.5em'})
        ]),
        
        # Risco Alto (71-100) - Possível incêndio natural
        html.Div([
            html.P([
                html.Span("🔴 ", style={'fontSize': '1.2em'}),
                html.Strong("Alto (71-100)")
            ], className="mb-0", style={'color': '#dc3545'}),
            html.P(
                stats['risco_alto']['descricao'],
                className="mb-0",
                style={'fontSize': '0.75em', 'fontStyle': 'italic', 'marginLeft': '1.5em'}
            ),
            html.P([
                f"{stats['risco_alto']['quantidade']:,} registros ",
                html.Span(
                    f"({stats['risco_alto']['percentual']:.1f}%)",
                    style={'fontWeight': 'bold'}
                )
            ], className="mb-0", style={'marginLeft': '1.5em'})
        ])
    ])
    
    return (
        figura,
        painel,
        btn_geral_color,
        btn_geral_outline,
        btn_estados_colors,
        btn_estados_outlines
    )