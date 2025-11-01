import plotly.express as px
import pandas as pd
from data.connection import connection_list

def gerar_lista_grafos():
    
    '''
    Gerar uma lista de gráficos de dispersão em mapa para cada DataFrame na lista, 
    guardando em variáveis para serem resgatadas no Dash.
    '''
    
    df_list = connection_list()
    grafos = []
    
    for df in df_list:
        
        if 'DataHora' in df.columns:
            df['DataHora'] = pd.to_datetime(df['Data'])
            ano = df['DataHora'].dt.year.mode()[0] if not df.empty else 'Desconhecido'
        else:
            ano = 'Desconhecido'
            
        fig = px.scatter_map(
            df,
            lat='Latitude',
            lon='Longitude',
            color='RiscoFogo',
            color_continuous_scale=px.colors.sequential.Turbo,
            hover_name="Estado",
            hover_data={
                'Municipio': True,
                'DataHora': True,
                'DiaSemChuva': True,
                'Precipitacao': True,
                'FRP': True,
                'Latitude': False,
                'Longitude': False
            },
            map_style='carto-positron',
            zoom=3,
            title=f"Risco de Fogo - {ano}"
        )
        
        # Centralizar o título
        fig.update_layout(
            title_x=0.5,
            title_xanchor='center'
        )
        
        grafos.append({
            'ano': int(ano) if ano != 'Desconhecido' else ano,
            'figura': fig,
            'df': df
        })

    grafos.sort(key=lambda x: x['ano'] if isinstance(x['ano'], int) else 0)
    
    return grafos