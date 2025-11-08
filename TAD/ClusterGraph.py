from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from TAD.Graph import Graph
from math import radians, sin, cos, sqrt, atan2

class ClusterGraph(Graph):
    '''
    Extensão do TAD Graph para análise de dados de incêndio no Cerrado.
    '''
    
    def __init__(self):
        super().__init__()
        self.__vertice_data: Dict[int, Dict[str, Any]] = {}
        self.__features_cache: Dict[int, Dict[str, float]] = {}
        
    # Getters para atributos privados
    def get_vertice_data_dict(self) -> Dict[int, Dict[str, Any]]:

        return self.__vertice_data.copy()
    
    def get_features_cache_dict(self) -> Dict[int, Dict[str, float]]:
        
        return self.__features_cache.copy()
    
    def limpar_cache(self) -> None:
        '''
        Limpa o cache de features
        '''
        
        self.__features_cache.clear()
        
    def add_vertice_com_dados(self, vertice_id: int, dados: Dict[str, Any]) -> None:
        '''
        Adiciona um vértice ao grafo com dados associados.
        '''
        
        self.add_ponto(vertice_id)
        self.__vertice_data[vertice_id] = dados
        
    def get_dados_vertice(self, vertice_id: int) -> Dict[str, Any]:
        '''
        Retorna os dados associados a um vértice.
        '''
        
        return self.__vertice_data.get(vertice_id, {}).copy()
    
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
        
        a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
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
        usar_espacial: bool = True,
        max_conexoes_por_vertice: int = 50,
        mostrar_progresso: bool = True
    ) -> None:
        '''
        Constrói o grafo a partir de um DataFrame de dados de incêndios.
        Conecta registros próximos espacialmente e/ou temporalmente.
        
        Otimizações para performance:
        - max_conexoes_por_vertice: Limita o número de conexões por vértice para reduzir complexidade
        - mostrar_progresso: Controla exibição de progresso (desabilitar para datasets grandes)
        '''
        
        if 'Data' not in df.columns and 'DataHora' in df.columns:
            df['Data'] = pd.to_datetime(df['DataHora'], errors='coerce')
        elif 'Data' in df.columns:
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')

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
        vertices = list(self.__vertice_data.keys())
        total_vertices = len(vertices)
        
        # Otimização: usar estrutura espacial para reduzir comparações
        if usar_espacial and total_vertices > 1000:
            self.__construir_arestas_otimizado(
                vertices, threshold_km, threshold_dias, 
                usar_temporal, usar_espacial, 
                max_conexoes_por_vertice, mostrar_progresso
            )
        else:
            self.__construir_arestas_simples(
                vertices, threshold_km, threshold_dias,
                usar_temporal, usar_espacial,
                max_conexoes_por_vertice, mostrar_progresso
            )
        
        num_vertices = len(self.get_pontos())
        num_arestas = len(self._graph)
        if mostrar_progresso:
            print(f'Grafo construído: {num_vertices} vértices, {num_arestas} arestas.')
    
    def __construir_arestas_simples(
        self, vertices: List[int], threshold_km: float, threshold_dias: int,
        usar_temporal: bool, usar_espacial: bool,
        max_conexoes: int, mostrar_progresso: bool
    ) -> None:
        '''
        Método privado para construir arestas de forma simples (para datasets pequenos)
        '''
        
        total_vertices = len(vertices)
        
        for i, v1 in enumerate(vertices):
            if mostrar_progresso and i % 1000 == 0:
                print(f'Processando vértice {i+1}/{total_vertices}')
                
            dados_v1 = self.__vertice_data[v1]
            conexoes_v1 = 0
            
            # Otimização: apenas verificar vértices posteriores para evitar duplicatas
            for v2 in vertices[i+1:]:
                if conexoes_v1 >= max_conexoes:
                    break
                    
                dados_v2 = self.__vertice_data[v2]
                
                conectar, peso = self.__verificar_conexao(
                    dados_v1, dados_v2, threshold_km, threshold_dias,
                    usar_temporal, usar_espacial
                )
                
                if conectar:
                    self.add_ponto_con(v1, v2,  peso)
                    conexoes_v1 += 1
    
    def __construir_arestas_otimizado(
        self, vertices: List[int], threshold_km: float, threshold_dias: int,
        usar_temporal: bool, usar_espacial: bool,
        max_conexoes: int, mostrar_progresso: bool
    ) -> None:
        '''
        Método privado para construir arestas de forma otimizada usando grid espacial.
        Reduz complexidade de O(n²) para aproximadamente O(n*k) onde k é o número médio de vizinhos.
        '''
        
        # Criar grid espacial para otimização
        grid_size = threshold_km / 111.0  # Aproximadamente 1 grau ~ 111 km
        grid: Dict[Tuple[int, int], List[int]] = {}
        
        # Indexar vértices no grid
        for v_id in vertices:
            dados = self.__vertice_data[v_id]
            grid_x = int(dados['latitude'] / grid_size)
            grid_y = int(dados['longitude'] / grid_size)
            grid_key = (grid_x, grid_y)
            
            if grid_key not in grid:
                grid[grid_key] = []
            grid[grid_key].append(v_id)
        
        total_vertices = len(vertices)
        processados = 0
        
        # Para cada vértice, verificar apenas células vizinhas no grid
        for v1 in vertices:
            if mostrar_progresso and processados % 5000 == 0:
                print(f'Processando vértice {processados+1}/{total_vertices}')

            dados_v1 = self.__vertice_data[v1]
            grid_x = int(dados_v1['latitude'] / grid_size)
            grid_y = int(dados_v1['longitude'] / grid_size)
            
            conexoes_v1 = 0
            
            # Verificar células vizinhas (3x3 grid)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    grid_key = (grid_x + dx, grid_y + dy)
                    if grid_key not in grid:
                        continue
                    
                    for v2 in grid[grid_key]:
                        if v2 <= v1 or conexoes_v1 >= max_conexoes:
                            continue
                        
                        dados_v2 = self.__vertice_data[v2]
                        
                        conectar, peso = self.__verificar_conexao(
                            dados_v1, dados_v2, threshold_km, threshold_dias,
                            usar_temporal, usar_espacial
                        )
                        
                        if conectar:
                            self.add_ponto_con(v1, v2,  peso)
                            conexoes_v1 += 1
            
            processados += 1
    
    def __verificar_conexao(
        self, dados_v1: Dict, dados_v2: Dict,
        threshold_km: float, threshold_dias: int,
        usar_temporal: bool, usar_espacial: bool
    ) -> Tuple[bool, float]:
        '''
        Método privado para verificar se dois vértices devem ser conectados.
        '''
        
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
        
        peso_medio = peso_total / componentes_peso if componentes_peso > 0 else 0.0
        return conectar, peso_medio
        
    def get_vizinhos(self, vertice_id: int) -> List[int]:
        '''
        Retorna a lista de vizinhos de um vértice.
        Usa o método herdado do Graph.
        '''
        
        return super().get_vizinhos(vertice_id)
    
    def get_peso(self, v1: int, v2: int) -> float:
        '''
        Retorna o peso da aresta entre dois vértices.
        Usa o método herdado do Graph.
        '''
        
        return super().get_peso(v1, v2)
        
    def calcular_grau(self, vertice_id: int) -> int:
        '''
        Calcula o grau de um vértice (número de vizinhos).
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
        
        riscos = [self.__vertice_data[v]['risco_fogo'] for v in vizinhos if v in self.__vertice_data]
        
        return np.mean(riscos) if riscos else 0.0
    
    def calcular_coeficiente_clustering(self, vertice_id: int) -> float:
        '''
        Calcula o coeficiente de clustering local de um vértice.
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
        
        n = len(self.get_pontos())
        
        if n <= 1:
            return 0.0
        
        grau = self.calcular_grau(vertice_id)
        return grau / (n - 1)
    
    def get_vizinhos_espaciais(self, vertice_id: int, raio_km: float) -> List[int]:
        '''
        Retorna vizinhos dentro de um raio geográfico especificado.
        '''
        
        if vertice_id not in self.__vertice_data:
            return []
        
        dados_origem = self.__vertice_data[vertice_id]
        vizinhos_raio = []
        
        for v_id, dados in self.__vertice_data.items():
            if v_id == vertice_id:
                continue
            
            dist = self.calcular_distancia_haversine(
                dados_origem['latitude'], dados_origem['longitude'],
                dados['latitude'], dados['longitude']
            )
            
            if dist <= raio_km:
                vizinhos_raio.append(v_id)
                
        return vizinhos_raio
    
    def get_vizinhos_temporais(self, vertice_id: int, janela_dias: int) -> List[int]:
        '''
        Retorna vizinhos dentro de uma janela temporal especificada.
        '''
        
        if vertice_id not in self.__vertice_data:
            return []
        
        dados_origem = self.__vertice_data[vertice_id]
        data_origem = dados_origem['data']
        
        if pd.isna(data_origem):
            return []
        
        vizinhos_temporais = []
        
        for v_id, dados in self.__vertice_data.items():
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
        if vertice_id in self.__features_cache:
            return self.__features_cache[vertice_id].copy()
        
        features = {
            'grafo_grau': self.calcular_grau(vertice_id),
            'grafo_peso_medio': self.calcular_peso_medio_arestas(vertice_id),
            'grafo_risco_medio_vizinhos': self.calcular_risco_medio_vizinhos(vertice_id),
            'grafo_coef_clustering': self.calcular_coeficiente_clustering(vertice_id),
            'grafo_centralidade_grau': self.calcular_centralidade_grau(vertice_id)
        }
        
        # Armazenar no cache
        self.__features_cache[vertice_id] = features
        
        return features.copy()
    
    def extrair_features_dataframe(self, df: pd.DataFrame, mostrar_progresso: bool = True) -> pd.DataFrame:
        '''
        Extrai features do grafo para todos os registros do DataFrame.
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
        total = len(df.index)
        if mostrar_progresso:
            print(f'Extraindo features do grafo para {total} registros...')
        
        for i, idx in enumerate(df.index):
            if mostrar_progresso and i % 5000 == 0 and i > 0:
                print(f'  Processado: {i}/{total} ({i/total*100:.1f}%)')
            
            if idx in self.__vertice_data:
                features = self.extrair_features_vertice(idx)
                for feature_name, feature_value in features.items():
                    df_copy.at[idx, feature_name] = feature_value
        
        if mostrar_progresso:
            print(f'  Concluído: {total}/{total} (100.0%)')
            print(f'Features extraídas: {feature_names}')
        
        return df_copy
    
    def identificar_regioes_criticas(self, percentil: float = 90) -> List[Tuple[int, float, float]]:
        '''
        Identifica regiões críticas baseadas em alta centralidade e alto risco.
        '''
        
        centralidades = []
        riscos = []
        
        for v_id in self.__vertice_data.keys():
            centralidade = self.calcular_centralidade_grau(v_id)
            risco = self.__vertice_data[v_id]['risco_fogo']
            centralidades.append(centralidade)
            riscos.append(risco)
        
        threshold_centralidade = np.percentile(centralidades, percentil)
        threshold_risco = np.percentile(riscos, percentil)
        
        regioes_criticas = []
        for v_id in self.__vertice_data.keys():
            centralidade = self.calcular_centralidade_grau(v_id)
            risco = self.__vertice_data[v_id]['risco_fogo']
            
            if centralidade >= threshold_centralidade and risco >= threshold_risco:
                regioes_criticas.append((v_id, centralidade, risco))
        
        # Ordenar por risco decrescente
        regioes_criticas.sort(key=lambda x: x[2], reverse=True)
        
        return regioes_criticas
    
    def calcular_risco_propagacao(self, vertice_id: int) -> float:
        '''
        Calcula um score de risco de propagação para um vértice.
        Combina risco próprio, risco dos vizinhos e conectividade.
        '''
        
        if vertice_id not in self.__vertice_data:
            return 0.0
        
        risco_proprio = self.__vertice_data[vertice_id]['risco_fogo']
        risco_vizinhos = self.calcular_risco_medio_vizinhos(vertice_id)
        grau = self.calcular_grau(vertice_id)
        
        # Score combinado: risco próprio + risco vizinhos + fator de conectividade
        score = (risco_proprio * 0.5) + (risco_vizinhos * 0.3) + (grau * 0.2)
        
        return score
    
    def __repr__(self) -> str:
        '''
        Representação em string do ClusterGraph
        '''
        
        num_vertices = len(self.get_pontos())
        num_arestas = len(self._graph)
        num_dados = len(self.__vertice_data)
        return (f'ClusterGraph: {num_vertices} vértices, {num_arestas} arestas | '
                f'Dados: {num_dados} registros')