from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from models.Graph import Graph
from math import radians, sin, cos, sqrt, atan2

class ClusterGraph(Graph):
    '''
    Extensão do TAD Graph para análise de dados de incêndio no Cerrado.
    '''
    
    def __init__(self):
        super().__init__()
        self._vertice_data: Dict[int, Dict[str, Any]] = {}
        self._features_cahce: Dict[int, Dict[str, float]] = {}
        
    def add_vertice_com_dados(self, vertice_id: int, dados: Dict[str, Any]) -> None:
        '''
        Adiciona um vértice ao grafo com dados associados.
        '''
        
        self.add_ponto(vertice_id)
        self._vertice_data[vertice_id] = dados
        
    def get_dados_vertice(self, vertice_id: int) -> Dict[str, Any]:
        '''
        Retorna os dados associados a um vértice.
        '''
        
        return self._vertice_data.get(vertice_id, {})
    
    @staticmethod
    def calcular_distancia_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        '''
        Calcula a distância em quilômetros entre duas coordenadas geográficas usando a fórmula de Haversine.
        '''
        
        R = 6371.0  # Raio da Terra em km
        
        lat1_rad = radians(lat1)
        lon1_rad = radians(lon1)
        lat2_rad = radians(lat2)
        lon2_rad = radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = sin(dlat / 2)**2  + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        distancia = R * c
        return distancia
    
    @staticmethod
    def calcular_diferenca_dias(data1: pd.Timestamp, data2: pd.Timestamp) -> int:
        '''
        Calcula a diferença em dias entre duas datas.
        '''
        
        return abs((data2 - data1).days)
    
    def construir_grafo_dataframe(
        self,
        df: pd.DataFrame,
        threshold_km: float = 50.0,
        threshold_dias: int = 7,
        usar_temporal: bool = True,
        usar_espacial: bool = True
    ) -> None:
        '''
        Constrói o grafo a partir de um DataFrame de dados de incêndios.
        Conecta registros próximos espacialmente e/ou temporalmente.
        
        Args:
            df: DataFrame com dados de incêndio
            threshold_km: Distância máxima em km para conexão espacial
            threshold_dias: Diferença máxima em dias para conexão temporal
            usar_temporal: Se True, considera proximidade temporal
            usar_espacial: Se True, considera proximidade espacial
        '''
        
        if 'Data' not in df.columns and 'DataHora' in df.columns:
            df['Data'] = pd.to_datetime(df['DataHora'], erros='coerce')
        elif 'Data' in df.columns:
            df['Data'] = pd.to_datetime(df['Data'], erros='coerce')
            
        # Adiciona vértices com dados
        for idx, row in df.iterrows():
            dados = {
                'latitude': row.get('Latitude', 0.0),
                'longitude': row.get('Longitude', 0.0),
                'data': row.get('Data'),
                'risco_fogo': row.get('RiscoFogo', 0),
                'precipitacao': row.get('Precipitacao', 0.0),
                'dias_sem_chuva': row.get('DiaSemChuva', 0),
                'estado': row.get('Estado', ''),
                'municipio': row.get('Municipio', '')
            }
            self.add_vertice_com_dados(idx, dados)
            
        # Adicionar arestas com base nas proximidades
        vertices = list(self._vertice_data.keys())
        total_vertices = len(vertices)
        
        for i, v1 in enumerate(vertices):
            if i % 1000 == 0:
                print(f"Processando vértice {i+1}/{total_vertices}")
                
            dados_v1 = self._vertice_data[v1]
            
            # Otimização: Apenas verificar vértices posteriores para evitar duplicatas
            for v2 in vertices[i+1:]:
                dados_v2 = self._vertice_data[v2]
                
                conectar = False
                peso_total = 0.0
                componentes_peso = 0
                
                # Verificar proximidade espacial
                if usar_espacial:
                    dist_km = self.calcular_distancia_haversine(
                        dados_v1['latitude'], dados_v1['longitude'],
                        dados_v2['latitude'], dados_v2['longitude']
                    )
                    
                    if dist_km <= threshold_km:
                        conectar = True
                        peso_total += dist_km / threshold_km
                        componentes_peso += 1
                        
                # Verificar proximidade temporal
                if usar_temporal and dados_v1['data'] is not pd.NaT and dados_v2['data'] is not pd.NaT:
                    diff_dias = self.calcular_diferenca_dias(dados_v1['data'], dados_v2['data'])
                    
                    if diff_dias <= threshold_dias:
                        conectar = True
                        peso_total += diff_dias / threshold_dias
                        componentes_peso += 1
                        
                # Adicionar aresta se critérios forem atendidos
                if conectar and componentes_peso > 0:
                    peso_medio = peso_total / componentes_peso
                    self.add_ponto_con(v1, v2, peso_medio)
                
        print(f'Grafo construído: {len(self)} vértices, {self.total_ponto_con()} arestas.')
        
    def calcular_grau(self, vertice_id: int) -> int:
        ''''
        Calcula o grau de um vértice (Número de vizinhos)
        '''
        
        return len(self.get_vizinhos(vertice_id))
    
    def calcular_peso_medio_arestas(self, vertice_id: int) -> float:
        '''
        Calcula o peso médio das arestas conectadas a um vértice.
        '''
        
        vizinhos = self.get_vizinhos(vertice_id)
        
        if not vizinhos:
            return 0.0
        
        pesos = [self.get_peso(vertice_id, v) for v in vizinhos] 
        return np.mean(pesos)
    
    def calcular_risco_medio_vizinhos(self, vertice_id: int) -> float:
        '''
        Calcula o risco médio de fogo dos vizinhos de um vértice.
        '''
        
        vizinhos = self.get_vizinhos(vertice_id)
        
        if not vizinhos:
            return 0.0
        
        riscos = [self._vertices_data[v]['risco_fogo'] for v in vizinhos if v in self._vertice_data]
        
        return np.mean(riscos) if riscos else 0.0
    
    def calcular_coeficiente_clustering(self, vertice_id: int) -> float:
        '''
        Calcula o coeficiente de clustering local de um vértice.
        Mede o quão conectados estão os vizinhos entre si.
        '''
        
        vizinhos = self.get_vizinhos(vertice_id)
        k = len(vizinhos)
        
        if k < 2:
            return 0.0
        
        # Contar as arestas entre os vizinhos
        arestas_entre_vizinhos = 0
        for i, v1 in enumerate(vizinhos):
            for v2 in vizinhos[i+1:]:
                if self.get_peso(v1, v2) > 0:
                    arestas_entre_vizinhos += 1
                    
        # Coeficiente = Arestas reais / Arestas possíveis
        arestas_possiveis = k * (k - 1) / 2

        return arestas_entre_vizinhos / arestas_possiveis if arestas_possiveis > 0 else 0.0
    
    def calcular_centralidade_grau(self, vertice_id: int) -> float:
        '''
        Calcula a centralidade de grau normalizada de um vértice.
        '''
        
        n = len(self)
        
        if n <= 1:
            return 0.0
        
        grau = self.calcular_grau(vertice_id)
        
        return grau / (n-1)
    
    def get_vizinhos_espaciais (self, vertice_id: int, raio_km: float) -> List[int]:
        '''
        Retorna vizinhos dentro de um rario geográfico especificado.
        '''
        
        if vertice_id not in self._vertice_data:
            return []
        
        dados_origen = self._vertice_data[vertice_id]
        vizinhos_raio = []
        
        for v_id, dados in self._vertice_data.items():
            if v_id == vertice_id:
                continue
            
            dist = self.calcular_distancia_haversine(
                dados_origen['latitude'], dados_origen['longitude'],
                dados['latitude'], dados['longitude']
            )
            
            if dist <= raio_km:
                vizinhos_raio.append(v_id)
                
        return vizinhos_raio
    
    def get_vizinhos_temporais(self, vertice_id: int, janela_dias: int) -> List[int]:
        '''
        Retorna vizinhos dentro de uma janela temporal especificada.
        '''
        
        if vertice_id not in self._vertice_data:
            return []
        
        dados_origem = self._vertice_data[vertice_id]
        data_origem = dados_origem['data']
        
        if pd.isna(data_origem):
            return []
        
        vizinhos_temporais = []
        
        for v_id, dados in self._vertice_data.items():
            if v_id == vertice_id or pd.isna(dados['data']):
                continue
            
            diff_dias = self.calcular_diferenca_dias(data_origem, dados['data'])
            
            if diff_dias <= janela_dias:
                vizinhos_temporais.append(v_id)
                
        return vizinhos_temporais
    
    def extrair_features_vertice(self, vertice_id: int) -> Dict[str, float]:
        '''
        Extrai todas as features derivadas do grafo para um vértice.
        '''
        
        # Verificar cache
        if vertice_id in self._features_cahce:
            return self._features_cahce[vertice_id]
        
        features = {
            'grafo_grau': self.calcular_grau(vertice_id),
            'grafo_peso_medio': self.calcular_peso_medio_arestas(vertice_id),
            'grafo_risco_medio_vizinhos': self.calcular_risco_medio_vizinhos(vertice_id),
            'grafo_coef_clustering': self.calcular_coeficiente_clustering(vertice_id),
            'grafo_centralidade_grau': self.calcular_centralidade_grau(vertice_id)
        }
        
        # Armazenar no cache
        self._features_cahce[vertice_id] = features
        
        return features
    
    def extrair_features_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Extrai features do grado para todos os registros do DataFrame.
        '''
        
        df_copy = df.copy()
        
        # Inicializar colunas de features
        feature_names = [
            'grafo_grau',
            'grafo_peso_medio',
            'grafo_risco_medio_vizinhos',
            'grafo_coef_clustering',
            'grafo_centralidade_grau'
        ]
        
        for feature in feature_names:
            df_copy[feature] = 0.0
            
        # Extrair features para cada vértice
        print('Extraindo features do grafo...')
        
        for idx in df.index:
            if idx in self._vertice_data:
                features = self.extrair_features_vertice(idx)
                
                for feature_name, feature_value in features.items():
                    df_copy.at[idx, feature_name] = feature_value
                    
        print(f'Features do grafo extraídas: {feature_names}')
        
        return df_copy
    
    def identificar_regioes_criticas(self, percentil: float = 90.0) -> List[Tuple[int, float, float]]:
        '''
        Identifica regiões críticas baseadas em alta centralidade e alto risco.
        '''
        
        centralidades = []
        riscos = []
        
        for v_id in self._vertice_data.keys():
            centralidade = self.calcular_centralidade_grau(v_id)
            risco = self._vertice_data[v_id]['risco_fogo']
            centralidades.append((centralidade))
            riscos.append((risco))
            
        threshold_centralidade = np.percentile(centralidades, percentil)
        theshold_risco = np.percentile(riscos, percentil)
        regioes_criticas = []
        
        for v_id in self._vertice_data.keys():
            centralidade = self.calcular_centralidade_grau(v_id)
            risco = self._vertice_data[v_id]['risco_fogo']
            
            if centralidade >= threshold_centralidade and risco >= theshold_risco:
                regioes_criticas.append((v_id, centralidade, risco))
                
        # Ordenar por risco descrescente
        regioes_criticas.sort(key=lambda x: x[2], reverse=True)
        
        return regioes_criticas
    
    def calcular_risco_propagacao(self, vertice_id: int) -> float:
        '''
        Calcula um score de risco de propagação para um vértice.
        Combina risco próprio, risco dos vizinhos e conectividade.
        '''
        
        if vertice_id not in self._vertice_data:
            return 0.0
        
        risco_proprio = self._vertice_data[vertice_id]['risco_fogo']
        risco_vizinhos = self.calcular_risco_medio_vizinhos(vertice_id)
        grau = self.calcular_grau(vertice_id)
        
        # Score combinado: risco próprio + risco vizinhos + fator de conectividade
        score = (risco_proprio * 0.5) + (risco_vizinhos * 0.3) + (grau * 0.2)
        
        return score
    
    def __repr__(self) -> str:
        '''
        Representação em string do ClusterGraph.
        '''
        
        return (f'ClusterGraph: {len(self)} vértices, {self.total_ponto_con()} arestas | '
                f'Dados: {len(self._vertice_data)} registros')
        
        
        
