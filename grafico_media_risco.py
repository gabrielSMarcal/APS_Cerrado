from data.connection import connection_list
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ESTADOS_CERRADO = [
    'BAHIA', 'DISTRITO FEDERAL', 'GOIÁS', 'MARANHÃO',
    'MATO GROSSO', 'MATO GROSSO DO SUL', 'MINAS GERAIS',
    'PARANÁ', 'PIAUÍ', 'RONDÔNIA', 'SÃO PAULO', 'TOCANTINS'
]

COORDENADAS_ESTADOS = {
    'BAHIA': (-12.5, -41.7, 6),
    'DISTRITO FEDERAL': (-15.8, -47.9, 9),
    'GOIÁS': (-15.8, -49.5, 6),
    'MARANHÃO': (-5.0, -45.0, 6),
    'MATO GROSSO': (-12.5, -55.5, 6),
    'MATO GROSSO DO SUL': (-20.5, -54.6, 6),
    'MINAS GERAIS': (-18.5, -44.5, 6),
    'PARANÁ': (-24.5, -51.5, 6),
    'PIAUÍ': (-7.5, -42.5, 6),
    'RONDÔNIA': (-11.0, -63.0, 6),
    'SÃO PAULO': (-22.5, -48.5, 6),
    'TOCANTINS': (-10.0, -48.0, 6)
}

def grafo_media_risco():
    """
    Calcula a média de RiscoFogo e a contagem de casos de fogo para cada estado do Cerrado, por ano,
    e plota o resultado em um scatter_map para cada ano, usando a contagem para o tamanho do ponto
    e exibindo a contagem como texto no marcador.
    """
    df_list = connection_list()

    df_coordenadas = pd.DataFrame({
        'Estado': list(COORDENADAS_ESTADOS.keys()),
        'Latitude': [v[0] for v in COORDENADAS_ESTADOS.values()],
        'Longitude': [v[1] for v in COORDENADAS_ESTADOS.values()]
    })
    
    center_lat = df_coordenadas['Latitude'].mean()
    center_lon = df_coordenadas['Longitude'].mean()
    
    for i, df_ano in enumerate(df_list):
        
        ano = 'Desconhecido'
        if 'DataHora' in df_ano.columns:
            df_ano['DataHora'] = pd.to_datetime(df_ano['DataHora'], errors='coerce')
            ano = df_ano['DataHora'].dt.year.mode()[0] if not df_ano.empty and not df_ano['DataHora'].isnull().all() else f'DataFrame {i+1}'
        elif 'Data' in df_ano.columns:
            df_ano['DataHora'] = pd.to_datetime(df_ano['Data'], errors='coerce')
            ano = df_ano['DataHora'].dt.year.mode()[0] if not df_ano.empty and not df_ano['DataHora'].isnull().all() else f'DataFrame {i+1}'
        else:
            ano = f'DataFrame {i+1}'

        if 'Estado' not in df_ano.columns:
            print(f"Coluna 'Estado' não encontrada no {ano}. Pulando.")
            continue
            
        df_ano['Estado'] = df_ano['Estado'].str.upper()

        df_cerrado_ano = df_ano[df_ano['Estado'].isin(ESTADOS_CERRADO)]
        
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
        
        df_plot_final = df_plot.dropna(subset=['MediaRiscoFogo'])
        df_plot_final['ContagemCasos'] = df_plot_final['ContagemCasos'].astype(int) 
        
        if df_plot_final.empty:
            print(f"Nenhum dado de RiscoFogo encontrado para os estados do Cerrado no {ano}. Pulando a plotagem.")
            continue
            
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
            marker=dict(
                sizemin=20
            ),
            selector=dict(type='scattermap') 
        )

        fig.show()

if __name__ == '__main__':
    grafo_media_risco()
