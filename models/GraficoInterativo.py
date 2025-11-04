import pandas as pd
import plotly.express as px
from typing import Optional, Dict, List
from data.connection import connection_list

class GraficoInterativo:
    '''
    Classe para gerenciar gráficos interativos de queimadas no Cerrado por ano.
    Encapsula dados, cache de figuras e geração de visualizações, facilitando a integração com Dash.
    '''
    
    def __init__(self):
        '''
        Inicializa a classe carregando os dados e gerando os gráficos.
        '''
        
        self.__grafos = []
        self.__ano_selecionado = None
        self.__figura_cache = {}
        
        # Carregar dados e gerar gráficos ao inicializar
        self.__carregar_e_gerar_grafos()
        
    def __carregar_e_gerar_grafos(self) -> None:
        '''
        Carrega os CSVs de cada ano e gera os gráficos correspondentes.
        Armazena as figuras em cache para acesso rápido.
        '''
        
        try:
            df_list = connection_list()
            
            if not df_list:
                raise RuntimeError("Nenhum DataFrame foi retornado pela conexão.")
            
            for df in df_list:
                # Extrair o ano do DataFrame
                if 'Data' in df.columns:
                    df['Data'] = pd.to_datetime(df['Data'])
                    df['DataHora'] = df['Data'].dt.date
                    ano = int(df['Data'].dt.year.mode()[0]) if not df.empty else 'Desconhecido'
                else:
                    ano = 'Desconhecido'
                
                # Validar colunas obrigatórias
                colunas_obrigatorias = [
                    'Estado', 'Municipio', 'Latitude', 'Longitude',
                    'RiscoFogo', 'Data', 'DiaSemChuva', 'Precipitacao', 'FRP'
                ]
                
                for col in colunas_obrigatorias:
                    if col not in df.columns:
                        raise ValueError(f"Coluna obrigatória '{col}' não encontrada no DataFrame do ano {ano}.")
                
                # Gerar figura para este ano
                fig = self.__gerar_figura(df, ano)
                
                # Armazenar no cache
                grafo_info = {
                    'ano': ano,
                    'df': df,
                    'figura': fig
                }
                
                self.__grafos.append(grafo_info)
                
                # Adicionar ao cache de figuras (apenas se ano for int)
                if isinstance(ano, int):
                    self.__figura_cache[ano] = fig
            
            # Ordenar grafos por ano
            self.__grafos.sort(key=lambda x: x['ano'] if isinstance(x['ano'], int) else 0)
            
            # Definir ano padrão (mais recente)
            if self.__grafos:
                self.__ano_selecionado = self.__grafos[-1]['ano']
                
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar e gerar gráficos: {e}")
    
    def __gerar_figura(self, df: pd.DataFrame, ano: int) -> px.scatter_map:
        '''
        Gera uma figura de mapa interativo para um DataFrame específico.
        
        Parâmetros:
            df (pd.DataFrame): DataFrame com os dados de queimadas
            ano (int): Ano correspondente aos dados
            
        Retorno:
            Figure do Plotly Express
        '''
        
        # Criar cópia para manipulação
        df_plot = df.copy()
        
        # Garantir que DataHora seja string no formato correto
        df_plot['DataHora'] = pd.to_datetime(df_plot['Data']).dt.strftime('%Y-%m-%d')
        
        # Criar o mapa de dispersão
        fig = px.scatter_map(
            df_plot,
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
            zoom=3,
            title=f'Risco de Fogo - {ano}'
        )
        
        # Centralizar o título
        fig.update_layout(
            title_x=0.5,
            title_xanchor='center'
        )
        
        return fig
    
    # Métodos públicos
    
    def obter_figura(self, ano: Optional[int] = None) -> px.scatter_map:
        '''
        Obtém a figura do mapa interativo para um ano específico.
        
        Parâmetros:
            ano (int): Ano desejado. Se None, retorna o ano atualmente selecionado.
            
        Retorno:
            Figure do Plotly Express
        '''
        
        # Se ano não especificado, usar o selecionado
        if ano is None:
            ano = self.__ano_selecionado
        
        # Buscar no cache
        if ano in self.__figura_cache:
            return self.__figura_cache[ano]
        
        # Se não encontrar, retornar a última figura disponível
        if self.__grafos:
            return self.__grafos[-1]['figura']
        
        return None
    
    def selecionar_ano(self, ano: int) -> None:
        '''
        Seleciona um ano específico para visualização.
        Se o ano não estiver disponível, mantém o ano atual.
        
        Parâmetros:
            ano (int): Ano a ser selecionado
        '''
        
        if ano in self.__figura_cache:
            self.__ano_selecionado = ano
    
    def obter_anos_disponiveis(self) -> List[int]:
        '''
        Retorna lista de anos disponíveis nos dados.
        
        Retorno:
            Lista de anos (int) ordenados
        '''
        
        return [g['ano'] for g in self.__grafos if isinstance(g['ano'], int)]
    
    def obter_lista_grafos(self) -> List[Dict]:
        '''
        Retorna a lista completa de grafos com suas informações.
        Compatível com a estrutura original de lista_grafos.py.
        
        Retorno:
            Lista de dicionários contendo 'ano', 'figura' e 'df'
        '''
        
        return self.__grafos
    
    def obter_estatisticas(self, ano: Optional[int] = None) -> Dict:
        '''
        Retorna estatísticas básicas para um ano específico.
        
        Parâmetros:
            ano (int): Ano desejado. Se None, usa o ano atualmente selecionado.
            
        Retorno:
            Dicionário com estatísticas do ano
        '''
        
        # Se ano não especificado, usar o selecionado
        if ano is None:
            ano = self.__ano_selecionado
        
        # Buscar DataFrame correspondente
        df = None
        for grafo in self.__grafos:
            if grafo['ano'] == ano:
                df = grafo['df']
                break
        
        if df is None:
            return {}
        
        # Calcular estatísticas
        risco_baixo = len(df[df['RiscoFogo'] <= 20])
        risco_medio = len(df[(df['RiscoFogo'] > 20) & (df['RiscoFogo'] <= 70)])
        risco_alto = len(df[df['RiscoFogo'] > 70])
        
        total = len(df)
        perc_baixo = (risco_baixo / total * 100) if total > 0 else 0
        perc_medio = (risco_medio / total * 100) if total > 0 else 0
        perc_alto = (risco_alto / total * 100) if total > 0 else 0
        
        return {
            'ano': ano,
            'total_registros': total,
            'estados_unicos': df['Estado'].nunique(),
            'municipios_unicos': df['Municipio'].nunique(),
            'risco_baixo': {
                'quantidade': risco_baixo,
                'percentual': perc_baixo,
                'descricao': 'Possível incêndio criminoso'
            },
            'risco_medio': {
                'quantidade': risco_medio,
                'percentual': perc_medio,
                'descricao': 'Situação intermediária'
            },
            'risco_alto': {
                'quantidade': risco_alto,
                'percentual': perc_alto,
                'descricao': 'Possível incêndio natural'
            }
        }
    
    '''
    PROPRIEDADES
    '''
    
    @property
    def ano_atual(self) -> Optional[int]:
        '''
        Retorna o ano atualmente selecionado.
        '''
        return self.__ano_selecionado
    
    @property
    def total_anos(self) -> int:
        '''
        Retorna o total de anos disponíveis.
        '''
        return len(self.__grafos)
    
    @property
    def ano_mais_recente(self) -> Optional[int]:
        '''
        Retorna o ano mais recente disponível.
        '''
        anos = self.obter_anos_disponiveis()
        return max(anos) if anos else None
    
    @property
    def ano_mais_antigo(self) -> Optional[int]:
        '''
        Retorna o ano mais antigo disponível.
        '''
        anos = self.obter_anos_disponiveis()
        return min(anos) if anos else None
    
    '''
    MÉTODOS ESPECIAIS
    '''
    
    def __repr__(self) -> str:
        '''
        Representação string da classe.
        '''
        return (
            f'GraficosInterativos('
            f'anos={self.total_anos}, '
            f'ano_atual={self.__ano_selecionado}, '
            f'range={self.ano_mais_antigo}-{self.ano_mais_recente})'
        )
    
    def __str__(self) -> str:
        '''
        String descritiva da classe.
        '''
        anos_disponiveis = ', '.join(map(str, self.obter_anos_disponiveis()))
        
        return (
            f'Gráficos Interativos de Queimadas\n'
            f'  Anos disponíveis: {anos_disponiveis}\n'
            f'  Ano selecionado: {self.__ano_selecionado}\n'
            f'  Total de anos: {self.total_anos}'
        )