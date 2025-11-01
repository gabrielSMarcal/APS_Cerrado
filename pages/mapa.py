from dash import dcc, html, Output, Input, State, ALL
import dash_bootstrap_components as dbc

from app import app
from models.mapa_interativo import MapaInterativo

# Carregando o CSV de previsão
mapa = MapaInterativo('data/treated_db/previsao_2026.csv')

estados_df = mapa.obter_estados_disponiveis()

# Mapeamento de estados para siglas
ESTADOS_SIGLAS = {
    'BAHIA': 'BA',
    'DISTRITO FEDERAL': 'DF',
    'GOIÁS': 'GO',
    'MARANHÃO': 'MA',
    'MATO GROSSO': 'MT',
    'MATO GROSSO DO SUL': 'MS',
    'MINAS GERAIS': 'MG',
    'PARANÁ': 'PR',
    'PIAUÍ': 'PI',
    'RONDÔNIA': 'RO',
    'SÃO PAULO': 'SP',
    'TOCANTINS': 'TO'
}

# Criar o mapa
layout = dbc.Container([
    # Título da página
    html.H3("Previsão de Queimadas no Cerrado - 2026", className="text-center my-4"),
    
    # Container principal com layout de duas colunas
    dbc.Row([
        # Coluna lateral - Filtros (reduzida)
        dbc.Col([
            html.Div([
                # Título da seção de filtros
                html.H5("Filtros", className="text-center mb-3"),
                
                # Botão para visualização geral (todos os estados)
                dbc.Button(
                    "🇧🇷 Brasil",
                    id="btn-todos-estados",
                    color="primary",
                    className="w-100 mb-3",
                    size="sm"
                ),
                
                html.Hr(),
                
                # Título da lista de estados
                html.H6("Estados:", className="text-center mb-2"),
                
                # Lista de botões para cada estado (com siglas)
                html.Div([
                    dbc.Button(
                        ESTADOS_SIGLAS.get(estado, estado[:2]),
                        id={'type': 'btn-estado', 'index': estado},
                        color="secondary",
                        outline=True,
                        className="w-100 mb-1",
                        size="sm",
                        style={'fontSize': '0.85rem', 'padding': '0.25rem'}
                    ) for estado in estados_df
                ])
                
            ], className="sticky-top", style={'top': '20px'})
        ], width=1, className="pe-2"),
        
        # Coluna principal - Mapa e Estatísticas
        dbc.Col([
            dbc.Row([
                # Mapa - 80% da largura
                dbc.Col([
                    dcc.Graph(
                        id='mapa-queimadas',
                        figure=mapa.obter_figura(),
                        style={'height': '85vh'}
                    )
                ], width=9),
                
                # Estatísticas - 20% da largura
                dbc.Col([
                    html.Div([
                        html.H5("Estatísticas", className="text-center mb-3"),
                        html.Div(id='painel-estatisticas', className="p-3 bg-light rounded")
                    ], className="sticky-top", style={'top': '20px'})
                ], width=3)
            ])
        ], width=11)
    ])
], fluid=True, className="px-4")


# CALLBACKS

@app.callback(
    [Output('mapa-queimadas', 'figure'),
     Output('painel-estatisticas', 'children'),
     Output('btn-todos-estados', 'color'),
     Output('btn-todos-estados', 'outline'),
     Output({'type': 'btn-estado', 'index': ALL}, 'color'),
     Output({'type': 'btn-estado', 'index': ALL}, 'outline')],
    [Input('btn-todos-estados', 'n_clicks'),
     Input({'type': 'btn-estado', 'index': ALL}, 'n_clicks')],
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
        ], className="mb-1", style={'fontSize': '0.9rem'}),
        html.P([
            html.Strong("Registros: "),
            f"{stats['total_registros']:,}"
        ], className="mb-1", style={'fontSize': '0.9rem'}),
        html.P([
            html.Strong("Municípios: "),
            f"{stats['municipios_unicos']}"
        ], className="mb-2", style={'fontSize': '0.9rem'}),
        
        html.Hr(className="my-2"),
        
        # Classificação de Risco
        html.H6("📈 Risco", className="text-center mb-2", style={'fontSize': '0.95rem'}),
        
        # Risco Baixo (0-20) - Possível incêndio criminoso
        html.Div([
            html.P([
                html.Span("🔴 ", style={'fontSize': '1em'}),
                html.Strong("Baixo (0-20)")
            ], className="mb-0", style={'color': '#dc3545', 'fontSize': '0.85rem'}),
            html.P(
                stats['risco_baixo']['descricao'],
                className="mb-0",
                style={'fontSize': '0.7em', 'fontStyle': 'italic', 'marginLeft': '1.2em'}
            ),
            html.P([
                f"{stats['risco_baixo']['quantidade']:,} ",
                html.Span(
                    f"({stats['risco_baixo']['percentual']:.1f}%)",
                    style={'fontWeight': 'bold'}
                )
            ], className="mb-2", style={'marginLeft': '1.2em', 'fontSize': '0.8rem'})
        ]),
        
        # Risco Médio (21-70)
        html.Div([
            html.P([
                html.Span("🟡 ", style={'fontSize': '1em'}),
                html.Strong("Médio (21-70)")
            ], className="mb-0", style={'color': '#ffc107', 'fontSize': '0.85rem'}),
            html.P(
                stats['risco_medio']['descricao'],
                className="mb-0",
                style={'fontSize': '0.7em', 'fontStyle': 'italic', 'marginLeft': '1.2em'}
            ),
            html.P([
                f"{stats['risco_medio']['quantidade']:,} ",
                html.Span(
                    f"({stats['risco_medio']['percentual']:.1f}%)",
                    style={'fontWeight': 'bold'}
                )
            ], className="mb-2", style={'marginLeft': '1.2em', 'fontSize': '0.8rem'})
        ]),
        
        # Risco Alto (71-100) - Possível incêndio natural
        html.Div([
            html.P([
                html.Span("🟢 ", style={'fontSize': '1em'}),
                html.Strong("Alto (71-100)")
            ], className="mb-0", style={'color': '#28a745', 'fontSize': '0.85rem'}),
            html.P(
                stats['risco_alto']['descricao'],
                className="mb-0",
                style={'fontSize': '0.7em', 'fontStyle': 'italic', 'marginLeft': '1.2em'}
            ),
            html.P([
                f"{stats['risco_alto']['quantidade']:,} ",
                html.Span(
                    f"({stats['risco_alto']['percentual']:.1f}%)",
                    style={'fontWeight': 'bold'}
                )
            ], className="mb-0", style={'marginLeft': '1.2em', 'fontSize': '0.8rem'})
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