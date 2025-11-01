import pandas as pd
import plotly.express as px
from typing import Optional, Dict, List, Tuple

from scipy import stats

class MapaInterativo:
    '''
    Classe para gerenciar mapa interativo de previsão de queimadas no Cerrado.
    Encapsula dados, filtros e geração de visualizações, facilitando a integração com Dash.
    '''
    
    # Constantes
    ESTADOS_CERRADO = [
        'BAHIA', 'DISTRITO FEDERAL', 'GOIÁS', 'MARANHÃO',
        'MATO GROSSO', 'MATO GROSSO DO SUL', 'MINAS GERAIS',
        'PARANÁ', 'PIAUÍ', 'RONDÔNIA', 'SÃO PAULO', 'TOCANTINS'
    ]
    
    # Coordenadas centrais de cada estado (lat, lon, zoom)
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
    
    def __init__(self, csv_path: str):
        
        self.__csv_path = csv_path
        self.__df_original = None
        self.__estado_selecionado = None
        self.__figura_cache = None
        
        # Carregar dados ao inicializar
        self.__carregar_dados()
        
    def __carregar_dados(self) -> None:
        '''
        Carrega o CSV de previsão em um DataFrame.
        '''
        
        try:
            self.__df_original = pd.read_csv(self.__csv_path)
            self.__df_filtrado = self.__df_original.copy()
            
            # Validar colunas esperadas
            colunas_obrigatorias = [
                'Estado', 'Municipio', 'Latitude', 'Longitude',
                'RiscoFogo', 'Data', 'DiaSemChuva', 'Precipitacao', 'FRP'
            ]
            
            # Validação das colunas
            for col in colunas_obrigatorias:
                if col not in self.__df_original.columns:
                    raise ValueError(f"Coluna obrigatória '{col}' não encontrada no CSV.")
                
            # Converter coluna Data para datetime
            self.__df_original['Data'] = pd.to_datetime(self.__df_original['Data'])
            self.__df_original['DataHora'] = self.__df_original['Data'].dt.date
            
            # Aplicar ao DF
            self.__df_filtrado['Data'] = pd.to_datetime(self.__df_filtrado['Data'])
            self.__df_filtrado['DataHora'] = self.__df_filtrado['Data'].dt.date
            
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar dados do CSV: {e}")
        
    def __validar_estado(self, estado: str) -> bool:
        '''
        Valida se o estado está na lista de estados do Cerrado.
        '''
        return estado in self.ESTADOS_CERRADO
    
    def __calcular_coordenadas_centrais(self) -> Tuple[float, float, int]:
        '''
        Calcula as coordenadas centrais (lat, lon, zoom) de um estado.
        
        Retorno:
            Tuple contendo (latitude, longitude, zoom)
        '''
        
        if self.__estado_selecionado and self.__estadio_selecionado in self.COORDENADAS_ESTADOS:
            return self.COORDENADAS_ESTADOS[self.__estado_selecionado]
        else:
            # Coordenadas padrão do Brasil
            return (-14.2350, -51.9253, 4)
        
    def __gerar_figuras(self) -> px.scatter_map:
        '''
        Gera a figura do mapa interativo com base no DataFrame filtrado.
        '''
        
        lat_central, lon_central, zoom = self.__calcular_coordenadas_centrais()
        
        # Determinar o título
        if self.__estado_selecionado:
            titulo = f'Previsão de Risco de Fogo em Cerrado no estado de {self.__estado_selecionado} - 2026'
        else:
            titulo = 'Previsão de Risco de Fogo no Cerrado - 2026'
            
        
        # Garantir que a coluna DataHora contenha apenas a data (sem horário) como string
        if 'Data' in self.__df_filtrado.columns:
            self.__df_filtrado['DataHora'] = pd.to_datetime(self.__df_filtrado['Data']).dt.strftime('%Y-%m-%d')
        elif 'DataHora' in self.__df_filtrado.columns:
            self.__df_filtrado['DataHora'] = pd.to_datetime(self.__df_filtrado['DataHora']).dt.strftime('%Y-%m-%d')

        # Criar o mapa
        fig = px.scatter_map(
            self.__df_filtrado,
            lat='Latitude',
            lon='Longitude',
            color='RiscoFogo',
            color_continuous_scale=px.colors.sequential.Turbo,
            hover_name='Estado',
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
            zoom=zoom,
            center={'lat': lat_central, 'lon': lon_central},
            title=titulo
        )
        
        # Centralizar o título
        fig.update_layout(
            title_x=0.5,
            title_xanchor='center'
        )
        
        return fig
    
    # Métodos públicos
    
    def filtrar_por_estado(self, estado: Optional[str] = None) -> None:
        '''
        Filtra o DataFrame pelo estado selecionado.
        
        Parâmetros:
            estado (str): Nome do estado para filtrar. Se None, remove o filtro.
        '''
        
        if estado is None:
            
            # Remover filtro
            self.__df_filtrado = self.__df_original.copy()
            self.__estado_selecionado = None
            self.__figura_cache = None
        else:
            
            # Validar estado
            if not self.__validar_estado(estado):
                raise ValueError(
                    f'Estado "{estado}" inválido.'
                    f' Escolha entre: {", ".join(self.ESTADOS_CERRADO)}'
                )
            
            # Aplicar filtro
            self.__estado_selecionado = estado
            self.__figura_cache = None
            
    def obter_figura(self, force_refresh: bool = False) -> px.scatter_map:
        '''
        Obtém a figura do mapa interativo.
        
        Parâmetros:
            force_refresh (bool): Se True, força a regeneração da figura.
        '''

        if self.__figura_cache is None or force_refresh:
            self.__figura_cache = self.__gerar_figura()

        return self.__figura_cache
    
    def obter_estados_disponiveis(self) -> List[str]:
        '''
        Retorna lista de estados disponíveis no DataFrame original.
        '''
        
        return sorted(self.__df_original['Estado'].unique().tolist())
    
    def obter_estatistica(self) -> Dict:
        '''
        Retorna estatísticas básicas do DataFrame filtrado.
        
        Classificação do RiscoFogo:
            - Baixo: 00-30
            - Médio: 31-70
            - Alto:  71-100
        '''

        df = self.__df_filtrado
        
        risco_baixo = len(df[df['RiscoFogo'] <= 30])
        risco_medio = len(df[(df['RiscoFogo'] > 30) & (df['RiscoFogo'] <= 70)])
        risco_alto = len(df[df['RiscoFogo'] > 70])
        
        # Calcular percentuais
        total = len(df)
        perc_baixo = (risco_baixo / total * 100) if total > 0 else 0
        perc_medio = (risco_medio / total * 100) if total > 0 else 0
        perc_alto = (risco_alto / total * 100) if total > 0 else 0
        
        return {
            'total_registros': total,
            'municipios': df['Municipio'].nunique(),
            risco_baixo: {
                'quantidade': risco_baixo,
                'percentual': perc_baixo,
                'descricao': 'Possível incêndio criminoso'
            },
            risco_medio: {
                'quantidade': risco_medio,
                'percentual': perc_medio,
                'descricao': 'Situação intermediária'
            },
            risco_alto: {
                'quantidade': risco_alto,
                'percentual': perc_alto,
                'descricao': 'Possível incêndio natural'
            }
        }
    
    def resetar_filtros(self) -> None:
        '''
        Reseta todos os filtros aplicados.
        '''
        
        self.filtrar_por_estado(None)
        
    '''
    PROPRIEDADES
    '''
    
    @property
    def estado_atual(self) -> Optional[str]:
        '''
        Retorna o estado atualmente selecionado.
        '''
        return self.__estado_selecionado
    
    @property
    def total_registros(self) -> int:
        '''
        Retorna o total de registros no DataFrame filtrado.
        '''
        return len(self.__df_filtrado)
    
    @property
    def total_registros_original(self) -> int:
        '''
        Retorna o total de registros no DataFrame original.
        '''
        return len(self.__df_original)
    
    '''
    MÉTODOS ESPECIAIS
    '''
    
    def __repr__(self) -> str:
        '''
        Representação string da classe.
        '''
        return (
            f'MapaInterativo('
            f'registros={self.total_registros}/{self.total_registros_original}, '
            f'estado={self.__estado_selecionado or "Todos"})'
        )
        
    def __str__(self) -> str:
        '''
        String descritiva da classe.
        '''
        
        return (
            f'Mapa Interativo de Queimadas\n'
            f'  Estado: {stats['estado_filtrado'] or 'Todos'}\n'
            f'  Registros: {stats['total_registros']}\n'
            f'  Municípios: {stats['municipios_unicos']}\n'
            f'  Risco Médio: {stats['risco_medio']:.2f}'
        )