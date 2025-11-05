from data.connection import connection_list
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from models.MapaInterativo import MapaInterativo

# Cache global para as figuras
_cache_figuras = {}
_cache_inicializado = False

def _inicializar_cache():
    """
    Inicializa o cache de figuras para todos os anos disponíveis.
    """
    
    global _cache_figuras, _cache_inicializado
    
    if _cache_inicializado:
        return
    
    df_list = connection_list()
    
    # Usar constantes da classe MapaInterativo
    ESTADOS_CERRADO = MapaInterativo.ESTADOS_CERRADO
    COORDENADAS_ESTADOS = MapaInterativo.COORDENADAS_ESTADOS

    df_coordenadas = pd.DataFrame({
        'Estado': list(COORDENADAS_ESTADOS.keys()),
        'Latitude': [v[0] for v in COORDENADAS_ESTADOS.values()],
        'Longitude': [v[1] for v in COORDENADAS_ESTADOS.values()]
    })
    
    center_lat = df_coordenadas['Latitude'].mean()
    center_lon = df_coordenadas['Longitude'].mean()
    
    for df in df_list:
        # Criar cópia logo no início para evitar warnings
        df_trabalho = df.copy()
        
        # Extrair ano
        if 'DataHora' in df_trabalho.columns:
            df_trabalho.loc[:, 'DataHora'] = pd.to_datetime(df_trabalho['DataHora'], errors='coerce')
            ano = int(df_trabalho['DataHora'].dt.year.mode()[0]) if not df_trabalho.empty and not df_trabalho['DataHora'].isnull().all() else None
        elif 'Data' in df_trabalho.columns:
            df_trabalho.loc[:, 'DataHora'] = pd.to_datetime(df_trabalho['Data'], errors='coerce')
            ano = int(df_trabalho['DataHora'].dt.year.mode()[0]) if not df_trabalho.empty and not df_trabalho['DataHora'].isnull().all() else None
        else:
            continue
        
        if ano is None or 'Estado' not in df_trabalho.columns:
            continue
        
        # Processar dados - garantir upper() em uma cópia
        df_trabalho.loc[:, 'Estado'] = df_trabalho['Estado'].str.upper()
        df_cerrado_ano = df_trabalho[df_trabalho['Estado'].isin(ESTADOS_CERRADO)].copy()
        
        if df_cerrado_ano.empty:
            continue
        
        df_analise_ano = df_cerrado_ano.groupby('Estado').agg(
            MediaRiscoFogo=('RiscoFogo', 'mean'),
            ContagemCasos=('RiscoFogo', 'count')
        ).reset_index()
        
        df_plot = pd.merge(
            df_coordenadas,
            df_analise_ano,
            on='Estado',
            how='left'
        )
        
        df_plot_final = df_plot.dropna(subset=['MediaRiscoFogo']).copy()
        
        if df_plot_final.empty:
            continue
        
        df_plot_final.loc[:, 'ContagemCasos'] = df_plot_final['ContagemCasos'].astype(int)
        
        # Gerar figura
        fig = px.scatter_map(
            df_plot_final,
            lat='Latitude',
            lon='Longitude',
            color='MediaRiscoFogo',
            color_continuous_scale=px.colors.sequential.Turbo,
            size='ContagemCasos', 
            size_max=40,
            hover_name='Estado',
            hover_data={
                'MediaRiscoFogo': ':.2f',
                'ContagemCasos': True,
                'Latitude': False,
                'Longitude': False
            },
            zoom=3, 
            map_style='carto-positron',
            center={'lat': center_lat, 'lon': center_lon},
            title=f'Média de Risco de Fogo e Contagem de Casos por Estado no Cerrado - Ano: {ano}'
        )

        fig.add_trace(
            go.Scattermap(
                lat=df_plot_final['Latitude'],
                lon=df_plot_final['Longitude'],
                mode='text',
                text=df_plot_final['ContagemCasos'].astype(str),
                textposition='middle center',
                textfont=dict(color='black', size=12, weight='bold'), 
                hoverinfo='none',
                showlegend=False
            )
        )

        fig.update_traces(
            marker=dict(sizemin=20),
            selector=dict(type='scattermap') 
        )
        
        fig.update_layout(
            title_x=0.5,
            title_xanchor='center'
        )
        
        # Armazenar no cache
        _cache_figuras[ano] = fig
    
    _cache_inicializado = True

def gerar_grafico_media_risco_por_ano(ano: int):
    """
    Gera um gráfico de média de risco para um ano específico.
    Usa cache para melhorar performance.
    """
    
    # Inicializar cache se necessário
    _inicializar_cache()
    
    # Retornar do cache
    return _cache_figuras.get(ano)

def grafo_media_risco():
    """
    Função original mantida para retrocompatibilidade.
    Gera e exibe gráficos para todos os anos.
    """
    
    _inicializar_cache()
    
    for ano in sorted(_cache_figuras.keys()):
        fig = _cache_figuras[ano]
        if fig:
            fig.show()
