from dash import html
import dash_bootstrap_components as dbc


def secao_pipeline_overview():
    '''
    Visão geral do pipeline de ML
    '''
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("🔄 Pipeline de Machine Learning", className="card-title mb-4"),
                    html.P([
                        "Nosso modelo segue um pipeline em ",
                        html.Strong("4 etapas principais"),
                        ", onde cada fase contribui para a qualidade final das previsões:"
                    ], className="mb-3"),
                    
                    # Lista das etapas
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("1️⃣ Preparação", className="text-primary"),
                                    html.P(
                                        "Criação de variáveis temporais, encoding de categóricas "
                                        "e normalização dos dados históricos.",
                                        className="small mb-0"
                                    )
                                ])
                            ], className="h-100 border-primary")
                        ], width=12, md=6, lg=3, className="mb-3"),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("2️⃣ Grafo", className="text-success"),
                                    html.P(
                                        "Construção de grafo espaço-temporal para capturar "
                                        "relações entre focos de incêndio próximos.",
                                        className="small mb-0"
                                    )
                                ])
                            ], className="h-100 border-success")
                        ], width=12, md=6, lg=3, className="mb-3"),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("3️⃣ Clustering", className="text-warning"),
                                    html.P(
                                        "K-means com k=12 para identificar padrões sazonais "
                                        "e agrupar meses similares.",
                                        className="small mb-0"
                                    )
                                ])
                            ], className="h-100 border-warning")
                        ], width=12, md=6, lg=3, className="mb-3"),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("4️⃣ Predição", className="text-info"),
                                    html.P(
                                        "Random Forest com 70 árvores para prever o RiscoFogo "
                                        "com alta precisão (R² > 0.93).",
                                        className="small mb-0"
                                    )
                                ])
                            ], className="h-100 border-info")
                        ], width=12, md=6, lg=3, className="mb-3")
                    ])
                ])
            ], className="shadow-sm mb-5")
        ], width=12)
    ])


def secao_grafo_espacotemporal():
    '''
    Explicação detalhada do grafo espaço-temporal
    '''
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("🕸️ Análise de Grafos Espaço-Temporal", 
                           className="card-title mb-4"),
                    
                    dbc.Row([
                        # Coluna esquerda: Explicação
                        dbc.Col([
                            html.H5("O que é um Grafo Espaço-Temporal?", 
                                   className="text-primary mb-3"),
                            html.P([
                                "Um grafo espaço-temporal é uma estrutura que conecta focos de "
                                "incêndio que ocorreram ",
                                html.Strong("próximos no espaço"),
                                " (até 50km de distância) e ",
                                html.Strong("próximos no tempo"),
                                " (até 14 dias de diferença). Essas conexões revelam padrões de "
                                "propagação e clusters de risco que não seriam capturados por "
                                "variáveis isoladas."
                            ], className="mb-4"),
                            
                            html.H5("Features Extraídas do Grafo", className="text-success mb-3"),
                            dbc.ListGroup([
                                dbc.ListGroupItem([
                                    html.Strong("Grau do Vértice: "),
                                    "Número de conexões que um foco tem com outros focos próximos. "
                                    "Indica se aquele ponto está em uma região de alta atividade."
                                ]),
                                dbc.ListGroupItem([
                                    html.Strong("Centralidade: "),
                                    "Mede quão central um nó é na rede. Identifica focos que "
                                    "podem ser pontos críticos de propagação."
                                ]),
                                dbc.ListGroupItem([
                                    html.Strong("Coeficiente de Clusterização: "),
                                    "Indica se os vizinhos de um nó também estão conectados entre si, "
                                    "revelando a presença de clusters densos de incêndios."
                                ])
                            ], className="mb-3")
                        ], width=12, lg=7),
                        
                        # Coluna direita: Visualização
                        dbc.Col([
                            dbc.Alert([
                                html.H6("📊 Impacto no Modelo", className="alert-heading"),
                                html.P([
                                    "As features de grafo ",
                                    html.Strong("melhoram significativamente"),
                                    " o desempenho do modelo, capturando padrões espaciais "
                                    "e temporais complexos que variáveis tradicionais não conseguem."
                                ], className="mb-2"),
                                html.Hr(),
                                html.P([
                                    "🎯 ",
                                    html.Strong("Resultado: "),
                                    "Modelos COM features de grafo apresentam R² superior e "
                                    "menor erro de predição."
                                ], className="mb-0 small")
                            ], color="info", className="mb-3"),
                            
                            # Placeholder informativo sobre o grafo
                            dbc.Alert([
                                html.H6("🕸️ Visualização de Grafo Espaço-Temporal", className="alert-heading mb-3"),
                                html.P([
                                    "O grafo conecta focos de incêndio que ocorreram ",
                                    html.Strong("próximos no espaço"),
                                    " (até 50km) e ",
                                    html.Strong("próximos no tempo"),
                                    " (até 7 dias), criando uma rede de relações complexas."
                                ], className="mb-2"),
                                html.Hr(),
                                html.P([
                                    "📊 ",
                                    html.Strong("Exemplo: "),
                                    "Um foco em Brasília (15°S, 48°W) no dia 15/08 conecta-se a ",
                                    "outro foco em Goiânia (16°S, 49°W) no dia 18/08, pois estão ",
                                    "a ~200km (dentro do raio) e 3 dias de diferença."
                                ], className="mb-0 small")
                            ], color="info", className="text-center")
                        ], width=12, lg=5)
                    ])
                ])
            ], className="shadow-sm mb-5")
        ], width=12)
    ])


def secao_clustering_kmeans():
    '''
    Explicação e visualização do clustering
    '''
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("📊 Clustering K-means (k=12)", className="card-title mb-4"),
                    
                    html.P([
                        "O clustering agrupa meses com características similares para identificar ",
                        html.Strong("padrões sazonais"),
                        " de risco de incêndio. Utilizamos o algoritmo K-means com ",
                        html.Strong("k=12 clusters"),
                        ", correspondendo aos 12 meses do ano, o que facilita a interpretação "
                        "dos padrões temporais."
                    ], className="mb-4"),
                    
                    html.H5("Por que k=12?", className="text-primary mb-3"),
                    html.P(
                        "A escolha de k=12 foi validada através de dois métodos complementares:",
                        className="mb-3"
                    ),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Img(
                                    src='/assets/avaliacao_outputs/graficos_clustering.png',
                                    style={'width': '100%'},
                                    alt='Visualização PCA dos Clusters',
                                    className="rounded shadow-sm"
                                )
                            ], className="mb-3"),
                            dbc.Alert([
                                html.Strong("🎨 Visualização PCA: "),
                                "Redução dimensional para 2D usando Análise de Componentes Principais. "
                                "Cores diferentes representam os 12 clusters, mostrando boa separação "
                                "entre os grupos sazonais."
                            ], color="light", className="small")
                        ], width=12, lg=6, className="mx-auto")
                    ], justify="center")
                ])
            ], className="shadow-sm mb-5")
        ], width=12)
    ])


def secao_random_forest():
    '''
    Explicação e visualização do Random Forest
    '''
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("🌲 Modelo Preditivo Random Forest", className="card-title mb-4"),
                    
                    html.P([
                        "O Random Forest é um ",
                        html.Strong("ensemble de árvores de decisão"),
                        " que combina múltiplas previsões para produzir um resultado mais robusto "
                        "e preciso. Cada árvore é treinada em uma amostra diferente dos dados, "
                        "reduzindo o risco de overfitting."
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Alert([
                                html.H6("⚙️ Parâmetros do Modelo", className="alert-heading"),
                                html.Ul([
                                    html.Li([
                                        html.Strong("70 árvores de decisão: "),
                                        "Equilíbrio entre performance e eficiência computacional"
                                    ]),
                                    html.Li([
                                        html.Strong("Profundidade máxima de 12: "),
                                        "Alinhado aos 12 clusters sazonais"
                                    ]),
                                    html.Li([
                                        html.Strong("Mínimo de 5 amostras por folha: "),
                                        "Evita overfitting e garante robustez"
                                    ])
                                ], className="mb-0")
                            ], color="info", className="mb-4")
                        ], width=12)
                    ]),
                    
                    html.H5("Importância das Features", className="text-success mb-3"),
                    html.P(
                        "O gráfico abaixo mostra quais variáveis mais influenciam as previsões. "
                        "Features de grafo estão destacadas em vermelho:",
                        className="mb-3"
                    ),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Img(
                                    src='/assets/avaliacao_outputs/feature_importance_rf.png',
                                    style={'width': '100%'},
                                    alt='Importância das Features',
                                    className="rounded shadow-sm"
                                )
                            ])
                        ], width=12, lg=6),
                        
                        dbc.Col([
                            html.Div([
                                html.Img(
                                    src='/assets/avaliacao_outputs/predicted_vs_real_scatter.png',
                                    style={'width': '100%'},
                                    alt='Predito vs Real',
                                    className="rounded shadow-sm"
                                )
                            ]),
                            html.P(
                                "Quanto mais próximos os pontos da linha vermelha (y=x), "
                                "melhor é a performance do modelo.",
                                className="text-muted small text-center mt-2"
                            )
                        ], width=12, lg=6)
                    ])
                ])
            ], className="shadow-sm mb-5")
        ], width=12)
    ])


def cards_metricas_melhorados():
    '''
    Cards de métricas com interpretações
    '''
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H5("MAE", className="card-title text-center mb-2"),
                        html.H3(id="mae-value", className="text-center text-primary mb-2"),
                        html.P("Erro Médio Absoluto", className="text-muted text-center small mb-2"),
                        html.Hr(className="my-2"),
                        html.P(
                            id="mae-interpretation", 
                            className="text-center small fst-italic mb-0",
                            style={'minHeight': '40px'}
                        )
                    ])
                ])
            ], className="shadow-sm h-100")
        ], width=12, md=6, lg=3, className="mb-3"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H5("RMSE", className="card-title text-center mb-2"),
                        html.H3(id="rmse-value", className="text-center text-warning mb-2"),
                        html.P("Raiz do Erro Quadrático", className="text-muted text-center small mb-2"),
                        html.Hr(className="my-2"),
                        html.P(
                            id="rmse-interpretation", 
                            className="text-center small fst-italic mb-0",
                            style={'minHeight': '40px'}
                        )
                    ])
                ])
            ], className="shadow-sm h-100")
        ], width=12, md=6, lg=3, className="mb-3"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H5("R²", className="card-title text-center mb-2"),
                        html.H3(id="r2-value", className="text-center text-success mb-2"),
                        html.P("Coeficiente de Determinação", className="text-muted text-center small mb-2"),
                        html.Hr(className="my-2"),
                        html.P(
                            id="r2-interpretation", 
                            className="text-center small fst-italic mb-0",
                            style={'minHeight': '40px'}
                        )
                    ])
                ])
            ], className="shadow-sm h-100")
        ], width=12, md=6, lg=3, className="mb-3"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H5("Acurácia (±10)", className="card-title text-center mb-2"),
                        html.H3(id="acc-value", className="text-center text-info mb-2"),
                        html.P("Predições na margem", className="text-muted text-center small mb-2"),
                        html.Hr(className="my-2"),
                        html.P(
                            id="acc-interpretation", 
                            className="text-center small fst-italic mb-0",
                            style={'minHeight': '40px'}
                        )
                    ])
                ])
            ], className="shadow-sm h-100")
        ], width=12, md=6, lg=3, className="mb-3")
    ])


def alerta_contexto_metricas():
    '''
    Alerta com contexto interpretativo das métricas
    '''
    
    return dbc.Row([
        dbc.Col([
            dbc.Alert([
                html.H6("💡 Interpretação dos Resultados", className="alert-heading mb-3"),
                html.P([
                    "Para referência, um ",
                    html.Strong("modelo baseline simples"),
                    " (como prever sempre a média histórica) teria um R² próximo de 0. "
                    "Nosso R² acima de ",
                    html.Strong("0.93"),
                    " indica que o modelo captura ",
                    html.Strong("mais de 93% da variabilidade dos dados"),
                    ", representando um desempenho ",
                    html.Strong("excelente"),
                    " para previsão de fenômenos ambientais complexos."
                ], className="mb-2"),
                html.P([
                    "A acurácia com margem de ±10 unidades mostra que ",
                    html.Strong("mais de 92% das previsões"),
                    " ficam dentro de uma faixa aceitável de erro, "
                    "tornando o modelo confiável para aplicações práticas de prevenção e monitoramento."
                ], className="mb-0")
            ], color="success", className="mb-4")
        ], width=12)
    ])