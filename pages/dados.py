from app import app
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
import os

import render.componentes_ml as componentes

# Carregar dados das métricas geradas
METRICS_PATH = './assets/avaliacao_outputs/metrics_continuas_por_ano.csv'
MARGIN_PATH = './assets/avaliacao_outputs/metrics_margem_por_ano.csv'

# Layout da página
layout = dbc.Container([
    # Store para evitar loops
    dcc.Store(id='dados-carregados', data=None),
    
    # Cabeçalho
    dbc.Row([
        dbc.Col([
            html.H2("🤖 Análise de Desempenho do Modelo de Machine Learning", 
                   className="text-center mb-3 mt-4"),
            html.P(
                "Explore o pipeline completo de ML que combina análise de grafos, "
                "clustering e Random Forest para prever riscos de incêndio no Cerrado.",
                className="text-center text-muted mb-4"
            ),
            html.Hr()
        ], width=12)
    ]),

    componentes.secao_pipeline_overview(),
    componentes.secao_grafo_espacotemporal(),
    componentes.secao_clustering_kmeans(),
    componentes.secao_random_forest(),

    # Título de métricas
    dbc.Row([
        dbc.Col([
            html.H3("📈 Métricas de Desempenho", className="mb-4 mt-5"),
        ], width=12)
    ]),
    
    # Seção de explicação do modelo
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📊 Sobre o Modelo de Previsão", className="card-title mb-3"),
                    html.P([
                        "O modelo utiliza técnicas de ",
                        html.Strong("Clustering (K-means com k=12)"),
                        " para agrupar anos com meses similares e prever o risco de incêndios no Cerrado Brasileiro. "
                        "A análise considera múltiplas variáveis ambientais e temporais, incluindo:"
                    ], className="mb-2"),
                    html.Ul([
                        html.Li("Dados históricos de focos de incêndio (INPE)"),
                        html.Li("Variáveis climáticas (temperatura, precipitação, umidade)"),
                        html.Li("Índices de vegetação e uso do solo"),
                        html.Li("Padrões temporais e sazonais"),
                    ], className="mb-3"),
                    html.P([
                        "O desempenho é avaliado através de métricas como ",
                        html.Strong("MAE (Mean Absolute Error)"),
                        ", ",
                        html.Strong("RMSE (Root Mean Square Error)"),
                        ", ",
                        html.Strong("R² (Coeficiente de Determinação)"),
                        " e ",
                        html.Strong("Acurácia com margem de erro"),
                        "."
                    ], className="text-muted")
                ])
            ], className="mb-4 shadow-sm")
        ], width=12)
    ]),
    
    # Seletor de ano
    dbc.Row([
        dbc.Col([
            html.Label("Selecione o Ano para Análise:", className="fw-bold"),
            dcc.Dropdown(
                id='ano-dropdown',
                placeholder="Selecione um ano...",
                className="mb-3",
                searchable=False,
                clearable=False
            )
        ], width=12, md=4)
    ]),
    
    # Explicação das métricas
    componentes.cards_metricas_melhorados(),
    componentes.alerta_contexto_metricas(),

    # Gráficos de métricas
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📈 Evolução das Métricas por Ano", className="card-title"),
                    html.P(
                        "Acompanhe como o erro médio do modelo varia ao longo dos anos.",
                        className="text-muted small"
                    ),
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
                    html.H5("🎯 Acurácia por Margem", className="card-title"),
                    html.P(
                        "Porcentagem de predições dentro de uma margem de erro aceitável (±10).",
                        className="text-muted small"
                    ),
                    dcc.Graph(
                        id='graph-acuracia-margem',
                        style={'height': '400px'},
                        config={'displayModeBar': False}
                    )
                ])
            ], className="shadow-sm", style={'height': '100%'})
        ], width=12, lg=6)
    ], className="mb-4"),
    
    # Tabela de métricas detalhadas
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📋 Métricas Detalhadas por Ano", className="card-title"),
                    html.P(
                        "Tabela completa com todas as métricas de desempenho do modelo.",
                        className="text-muted small mb-3"
                    ),
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
     Output('acc-value', 'children'),
     Output('mae-interpretation', 'children'),
     Output('rmse-interpretation', 'children'),
     Output('r2-interpretation', 'children'),
     Output('acc-interpretation', 'children')],
    Input('ano-dropdown', 'value')
)
def update_metrics_cards(ano_selecionado):
    '''
    Atualiza os cards de métricas e suas interpretações
    '''
    
    if ano_selecionado is None or not os.path.exists(METRICS_PATH):
        return "-", "-", "-", "-", "-", "-", "-", "-"
    
    df_metrics = pd.read_csv(METRICS_PATH)
    df_margin = pd.read_csv(MARGIN_PATH) if os.path.exists(MARGIN_PATH) else pd.DataFrame()
    
    row_metrics = df_metrics[df_metrics['Ano'] == ano_selecionado]
    
    if row_metrics.empty:
        return "-", "-", "-", "-", "-", "-", "-", "-"
    
    # Extrair valores numéricos ANTES de formatar
    mae_num = row_metrics['MAE'].values[0]
    rmse_num = row_metrics['RMSE'].values[0]
    r2_num = row_metrics['R2'].values[0]
    
    # Formatar para exibição nos cards
    mae_str = f"{mae_num:.2f}"
    rmse_str = f"{rmse_num:.2f}"
    r2_str = f"{r2_num:.3f}"
    
    # Processar acurácia
    if not df_margin.empty:
        row_margin = df_margin[df_margin['Ano'] == ano_selecionado]
        if not row_margin.empty:
            acc_num = row_margin['accuracy_margin'].values[0] * 100
            acc_str = f"{acc_num:.1f}%"
        else:
            acc_str = "-"
            acc_num = 0
    else:
        acc_str = "-"
        acc_num = 0
    
    # Usar valores numéricos para cálculos e comparações
    mae_interp = f"Erro médio de {mae_num:.2f} unidades de risco"
    
    # Comparar valores numéricos
    if rmse_num > mae_num * 1.5:
        rmse_interp = "Penaliza erros grandes"
    else:
        rmse_interp = "Erros consistentes"
    
    # Interpretar R²
    if r2_num > 0.9:
        r2_interp = f"Explica {r2_num*100:.1f}% da variação - Excelente!"
    elif r2_num > 0.7:
        r2_interp = f"Explica {r2_num*100:.1f}% da variação - Bom"
    else:
        r2_interp = f"Explica {r2_num*100:.1f}% da variação"
    
    # Interpretar acurácia
    if acc_str != "-":
        if acc_num > 90:
            acc_interp = "Excelente precisão!"
        elif acc_num > 80:
            acc_interp = "Boa precisão"
        else:
            acc_interp = "Precisão aceitável"
    else:
        acc_interp = "Precisão do modelo"
    
    return (
        mae_str, rmse_str, r2_str, acc_str,
        mae_interp, rmse_interp, r2_interp, acc_interp
    )


@app.callback(
    Output('graph-metricas-ano', 'figure'),
    Input('dados-carregados', 'data')
)

def update_graph_metricas(dados_carregados):
    '''
    Gerar gráfico de métricas ao longo dos anos
    '''
    
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
            line=dict(color='#3498db', width=2),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=df['Ano'], 
            y=df['RMSE'], 
            mode='lines+markers', 
            name='RMSE', 
            line=dict(color='#f39c12', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            xaxis_title="Ano",
            yaxis_title="Erro",
            hovermode='x unified',
            template='plotly_white',
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
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
    '''
    Gerar gráfico de acurácia por margem ao longo dos anos
    '''
    
    if not dados_carregados or not os.path.exists(MARGIN_PATH):
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
            marker=dict(size=10),
            fill='tozeroy',
            fillcolor='rgba(39, 174, 96, 0.1)'
        ))
        
        fig.update_layout(
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
    Output('tabela-metricas', 'children'),
    Input('dados-carregados', 'data')
)

def update_table(dados_carregados):
    '''
    Gerar tabela de métricas detalhadas por ano
    '''
    
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
    
    # Ordenar por ano decrescente
    df_full = df_full.sort_values('Ano', ascending=False)
    
    # Criar tabela Bootstrap
    table_header = [
        html.Thead(html.Tr([
            html.Th("Ano", className="text-center"),
            html.Th("MAE", className="text-center"),
            html.Th("RMSE", className="text-center"),
            html.Th("R²", className="text-center"),
            html.Th("Acurácia (±10)", className="text-center"),
            html.Th("Amostras", className="text-center")
        ]), className="table-light")
    ]
    
    rows = []
    for _, row in df_full.iterrows():
        acc_val = f"{row['accuracy_margin']:.1f}%" if pd.notna(row.get('accuracy_margin')) else "-"
        
        # Adicionar classe de cor baseado na acurácia
        row_class = ""
        if pd.notna(row.get('accuracy_margin')):
            if row['accuracy_margin'] >= 80:
                row_class = "table-success"
            elif row['accuracy_margin'] >= 60:
                row_class = "table-warning"
            else:
                row_class = "table-danger"
        
        rows.append(html.Tr([
            html.Td(int(row['Ano']), className="text-center fw-bold"),
            html.Td(f"{row['MAE']:.2f}", className="text-center"),
            html.Td(f"{row['RMSE']:.2f}", className="text-center"),
            html.Td(f"{row['R2']:.3f}", className="text-center"),
            html.Td(acc_val, className="text-center fw-bold"),
            html.Td(f"{int(row['n']):,}", className="text-center text-muted")
        ], className=row_class))
    
    table_body = [html.Tbody(rows)]
    
    return dbc.Table(
        table_header + table_body, 
        bordered=True, 
        hover=True, 
        responsive=True, 
        striped=True,
        className="mb-0"
    )