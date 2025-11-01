from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from app import app
import pandas as pd
import os

# Carregar dados das métricas geradas
METRICS_PATH = './avaliacao_outputs/metrics_continuas_por_ano.csv'
MARGIN_PATH = './avaliacao_outputs/metrics_margem_por_ano.csv'

# Layout da página
layout = dbc.Container([
    # Store para evitar loops
    dcc.Store(id='dados-carregados', data=None),
    
    # Cabeçalho
    dbc.Row([
        dbc.Col([
            html.H2("Análise de Desempenho do Modelo", className="text-center mb-4 mt-4"),
            html.Hr()
        ], width=12)
    ]),
    
    # Seção de métricas resumidas
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Métricas de Avaliação", className="card-title text-center"),
                    html.P("Análise do desempenho do modelo de Machine Learning", 
                           className="text-muted text-center")
                ])
            ], className="mb-4 shadow-sm")
        ], width=12)
    ]),
    
    # Seletor de ano
    dbc.Row([
        dbc.Col([
            html.Label("Selecione o Ano:", className="fw-bold"),
            dcc.Dropdown(
                id='ano-dropdown',
                placeholder="Selecione um ano...",
                className="mb-3",
                searchable=False,
                clearable=False
            )
        ], width=12, md=4)
    ]),
    
    # Cards com métricas principais
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("MAE", className="card-title text-center"),
                    html.H3(id="mae-value", className="text-center text-primary"),
                    html.P("Erro Médio Absoluto", className="text-muted text-center small")
                ])
            ], className="shadow-sm")
        ], width=12, md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("RMSE", className="card-title text-center"),
                    html.H3(id="rmse-value", className="text-center text-warning"),
                    html.P("Raiz do Erro Quadrático Médio", className="text-muted text-center small")
                ])
            ], className="shadow-sm")
        ], width=12, md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("R²", className="card-title text-center"),
                    html.H3(id="r2-value", className="text-center text-success"),
                    html.P("Coeficiente de Determinação", className="text-muted text-center small")
                ])
            ], className="shadow-sm")
        ], width=12, md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Acurácia (±10)", className="card-title text-center"),
                    html.H3(id="acc-value", className="text-center text-info"),
                    html.P("Predições dentro da margem", className="text-muted text-center small")
                ])
            ], className="shadow-sm")
        ], width=12, md=3)
    ], className="mb-4"),
    
    
    # Gráficos de métricas
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Evolução das Métricas por Ano", className="card-title"),
                    dcc.Graph(
                        id='graph-metricas-ano',
                        style={'height': '400px'},
                        config={'displayModeBar': False}
                    )
                ])
            ], className="shadow-sm", style={'height': '100%'})
        ], width=12, lg=6, className="mb-3 mb-lg-0"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Acurácia por Margem", className="card-title"),
                    dcc.Graph(
                        id='graph-acuracia-margem',
                        style={'height': '400px'},
                        config={'displayModeBar': False}
                    )
                ])
            ], className="shadow-sm", style={'height': '100%'})
        ], width=12, lg=6)
    ], className="mb-4"),
    
    # Mapas interativos
    dbc.Row([
        dbc.Col([
            html.H4("Visualização Geoespacial - 2025", className="mb-3"),
            dbc.Tabs([
                dbc.Tab(label="Mapa de Acertos/Erros", tab_id="tab-mapa-erros"),
                dbc.Tab(label="Heatmap de Erros", tab_id="tab-heatmap")
            ], id="tabs-mapas", active_tab="tab-mapa-erros"),
            html.Div(id="content-mapas", className="mt-3")
        ], width=12)
    ], className="mb-4"),
    
    # Tabela de métricas detalhadas
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Métricas Detalhadas por Ano", className="card-title"),
                    html.Div(id='tabela-metricas')
                ])
            ], className="shadow-sm")
        ], width=12)
    ], className="mb-5")
    
], fluid=True, className="py-3")


# Callback para carregar dados uma única vez
@app.callback(
    Output('dados-carregados', 'data'),
    Input('dados-carregados', 'id')
)
def carregar_dados_iniciais(_):
    return True


# Callbacks para atualizar os componentes
@app.callback(
    [Output('ano-dropdown', 'options'),
     Output('ano-dropdown', 'value')],
    Input('dados-carregados', 'data')
)
def populate_anos(dados_carregados):
    if dados_carregados and os.path.exists(METRICS_PATH):
        df = pd.read_csv(METRICS_PATH)
        anos = sorted(df['Ano'].unique())
        options = [{'label': str(ano), 'value': ano} for ano in anos]
        return options, anos[-1] if anos else None
    return [], None


@app.callback(
    [Output('mae-value', 'children'),
     Output('rmse-value', 'children'),
     Output('r2-value', 'children'),
     Output('acc-value', 'children')],
    Input('ano-dropdown', 'value')
)
def update_metrics_cards(ano_selecionado):
    if ano_selecionado is None or not os.path.exists(METRICS_PATH):
        return "-", "-", "-", "-"
    
    df_metrics = pd.read_csv(METRICS_PATH)
    df_margin = pd.read_csv(MARGIN_PATH) if os.path.exists(MARGIN_PATH) else pd.DataFrame()
    
    row_metrics = df_metrics[df_metrics['Ano'] == ano_selecionado]
    
    if row_metrics.empty:
        return "-", "-", "-", "-"
    
    mae = f"{row_metrics['MAE'].values[0]:.2f}"
    rmse = f"{row_metrics['RMSE'].values[0]:.2f}"
    r2 = f"{row_metrics['R2'].values[0]:.3f}"
    
    if not df_margin.empty:
        row_margin = df_margin[df_margin['Ano'] == ano_selecionado]
        acc = f"{row_margin['accuracy_margin'].values[0]*100:.1f}%" if not row_margin.empty else "-"
    else:
        acc = "-"
    
    return mae, rmse, r2, acc


@app.callback(
    Output('graph-metricas-ano', 'figure'),
    Input('dados-carregados', 'data')
)
def update_graph_metricas(dados_carregados):
    import plotly.graph_objs as go
    
    if not dados_carregados or not os.path.exists(METRICS_PATH):
        fig = go.Figure()
        fig.add_annotation(
            text="Dados não disponíveis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
    
    try:
        df = pd.read_csv(METRICS_PATH)
        
        if df.empty or 'Ano' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Estrutura de dados inválida",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=400
            )
            return fig
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Ano'], 
            y=df['MAE'], 
            mode='lines+markers', 
            name='MAE', 
            line=dict(color='#3498db', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df['Ano'], 
            y=df['RMSE'], 
            mode='lines+markers', 
            name='RMSE', 
            line=dict(color='#f39c12', width=2)
        ))
        
        fig.update_layout(
            title="MAE e RMSE por Ano",
            xaxis_title="Ano",
            yaxis_title="Erro",
            hovermode='x unified',
            template='plotly_white',
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
        
    except Exception as e:
        print(f"Erro ao gerar gráfico de métricas: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Erro ao carregar dados: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color="red")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig


@app.callback(
    Output('graph-acuracia-margem', 'figure'),
    Input('dados-carregados', 'data')
)
def update_graph_acuracia(dados_carregados):
    import plotly.graph_objs as go
    
    if not dados_carregados or not os.path.exists(MARGIN_PATH):
        # Retornar figura vazia com mensagem
        fig = go.Figure()
        fig.add_annotation(
            text="Dados não disponíveis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
    
    try:
        df = pd.read_csv(MARGIN_PATH)
        
        if df.empty or 'Ano' not in df.columns or 'accuracy_margin' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Estrutura de dados inválida",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=400
            )
            return fig
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Ano'], 
            y=df['accuracy_margin']*100,
            mode='lines+markers',
            name='Acurácia',
            line=dict(color='#27ae60', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Acurácia (±10) por Ano",
            xaxis_title="Ano",
            yaxis_title="Acurácia (%)",
            yaxis=dict(range=[0, 100]),
            hovermode='x unified',
            template='plotly_white',
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
        
    except Exception as e:
        print(f"Erro ao gerar gráfico de acurácia: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Erro ao carregar dados: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color="red")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig


@app.callback(
    Output('content-mapas', 'children'),
    Input('tabs-mapas', 'active_tab')
)
def render_map_content(active_tab):
    if active_tab == "tab-mapa-erros":
        map_path = 'avaliacao_outputs/map_2025_acertos_erros.html'
        if os.path.exists(map_path):
            with open(map_path, 'r', encoding='utf-8') as f:
                map_html = f.read()
            return html.Iframe(srcDoc=map_html, style={'width': '100%', 'height': '600px', 'border': 'none'})
        return html.Div("Mapa não disponível", className="text-muted text-center p-5")
    
    elif active_tab == "tab-heatmap":
        heat_path = 'avaliacao_outputs/heatmap_erro_agregado.html'
        if os.path.exists(heat_path):
            with open(heat_path, 'r', encoding='utf-8') as f:
                heat_html = f.read()
            return html.Iframe(srcDoc=heat_html, style={'width': '100%', 'height': '600px', 'border': 'none'})
        return html.Div("Heatmap não disponível", className="text-muted text-center p-5")
    
    return html.Div()


@app.callback(
    Output('tabela-metricas', 'children'),
    Input('dados-carregados', 'data')
)
def update_table(dados_carregados):
    if not dados_carregados or not os.path.exists(METRICS_PATH):
        return html.P("Dados não disponíveis", className="text-muted")
    
    df_metrics = pd.read_csv(METRICS_PATH)
    
    if os.path.exists(MARGIN_PATH):
        df_margin = pd.read_csv(MARGIN_PATH)
        df_full = pd.merge(df_metrics, df_margin[['Ano', 'accuracy_margin']], on='Ano', how='left')
        df_full['accuracy_margin'] = (df_full['accuracy_margin'] * 100).round(1)
    else:
        df_full = df_metrics.copy()
        df_full['accuracy_margin'] = None
    
    # Criar tabela Bootstrap
    table_header = [
        html.Thead(html.Tr([
            html.Th("Ano"),
            html.Th("MAE"),
            html.Th("RMSE"),
            html.Th("R²"),
            html.Th("Acurácia (%)"),
            html.Th("Amostras")
        ]))
    ]
    
    rows = []
    for _, row in df_full.iterrows():
        acc_val = f"{row['accuracy_margin']:.1f}%" if pd.notna(row.get('accuracy_margin')) else "-"
        rows.append(html.Tr([
            html.Td(int(row['Ano'])),
            html.Td(f"{row['MAE']:.2f}"),
            html.Td(f"{row['RMSE']:.2f}"),
            html.Td(f"{row['R2']:.3f}"),
            html.Td(acc_val),
            html.Td(int(row['n']))
        ]))
    
    table_body = [html.Tbody(rows)]
    
    return dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True, striped=True)

