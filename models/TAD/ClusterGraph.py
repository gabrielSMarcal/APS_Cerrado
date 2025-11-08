from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from models.TAD.Graph import Graph
from math import radians, sin, cos, sqrt, atan2

class ClusterGraph(Graph):
    
    def __init__(self):
        super().__init__()
        self.__vertice_data: Dict[int, Dict[str, Any]] = {}
        self.__features_cache: Dict[int, Dict[str, float]] = {}
        
    def get_vertice_data_dict(self) -> Dict[int, Dict[str, Any]]:
        return self.__vertice_data.copy()
    
    def get_features_cache_dict(self) -> Dict[int, Dict[str, float]]:
        return self.__features_cache.copy()
    
    def limpar_cache(self) -> None:
        self.__features_cache.clear()
        
    def add_vertice_com_dados(self, vertice_id: int, dados: Dict[str, Any]) -> None:
        self.add_ponto(vertice_id)
        self.__vertice_data[vertice_id] = dados
        
    def get_dados_vertice(self, vertice_id: int) -> Dict[str, Any]:
        return self.__vertice_data.get(vertice_id, {}).copy()
    
    @staticmethod
    def calcular_distancia_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        lat1_rad = radians(lat1)
        lon1_rad = radians(lon1)
        lat2_rad = radians(lat2)
        lon2_rad = radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c
    
    @staticmethod
    def calcular_diferenca_dias(data1: pd.Timestamp, data2: pd.Timestamp) -> int:
        return abs((data2 - data1).days)
    
    def construir_grafo_dataframe(
        self,
        df: pd.DataFrame,
        threshold_km: float = 50.0,
        threshold_dias: int = 7,
        usar_temporal: bool = True,
        usar_espacial: bool = True,
        max_conexoes_por_vertice: int = 10,
        mostrar_progresso: bool = True,
        grid_size_km: float = 50.0,
        janela_temporal_dias: int = 14
    ) -> None:
        
        if 'Data' not in df.columns and 'DataHora' in df.columns:
            df['Data'] = pd.to_datetime(df['DataHora'], errors='coerce')
        elif 'Data' in df.columns:
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')

        if mostrar_progresso:
            print(f'Agregando {len(df)} registros em células espaciais-temporais...')
            print(f'Parâmetros: grid_size={grid_size_km}km, janela_temporal={janela_temporal_dias}dias')
        
        df_agregado = self.__agregar_dados(df, grid_size_km, janela_temporal_dias)
        
        if mostrar_progresso:
            reducao = (1 - len(df_agregado) / len(df)) * 100
            print(f'Reduzido para {len(df_agregado)} vértices ({reducao:.1f}% de redução)')

        for idx, row in df_agregado.iterrows():
            dados = {
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'data': row['data'],
                'risco_fogo': row['risco_fogo'],
                'precipitacao': row['precipitacao'],
                'dias_sem_chuva': row['dias_sem_chuva'],
                'estado': row['estado'],
                'municipio': row['municipio'],
                'num_registros': row['num_registros']
            }
            self.add_vertice_com_dados(idx, dados)
            
        vertices = list(self.__vertice_data.keys())
        self.__construir_arestas_agregadas(
            vertices, threshold_km, threshold_dias,
            usar_temporal, usar_espacial,
            max_conexoes_por_vertice, mostrar_progresso
        )
        
        if mostrar_progresso:
            print(f'Grafo: {len(self.get_pontos())} vértices, {self.total_ponto_con()} arestas')
    
    def __agregar_dados(
        self, 
        df: pd.DataFrame, 
        grid_size_km: float, 
        janela_temporal_dias: int
    ) -> pd.DataFrame:
        
        df = df.copy()
        grid_size_deg = grid_size_km / 111.0
        
        df['grid_lat'] = (df['Latitude'] / grid_size_deg).astype(int)
        df['grid_lon'] = (df['Longitude'] / grid_size_deg).astype(int)
        df['periodo_temporal'] = (df['Data'] - df['Data'].min()).dt.days // janela_temporal_dias
        
        agg_dict = {
            'Latitude': 'mean',
            'Longitude': 'mean',
            'Data': 'mean',
            'Precipitacao': 'mean',
            'DiaSemChuva': 'mean',
            'Estado': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
            'Municipio': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        }
        
        if 'RiscoFogo' in df.columns:
            agg_dict['RiscoFogo'] = 'mean'
        
        agregacao = df.groupby(['grid_lat', 'grid_lon', 'periodo_temporal']).agg(agg_dict).reset_index()
        
        agregacao['num_registros'] = df.groupby(['grid_lat', 'grid_lon', 'periodo_temporal']).size().values
        
        colunas_base = ['grid_lat', 'grid_lon', 'periodo_temporal', 'latitude', 'longitude', 
                        'data', 'precipitacao', 'dias_sem_chuva', 'estado', 'municipio']
        
        if 'RiscoFogo' in df.columns:
            colunas_base.insert(6, 'risco_fogo')
        
        colunas_base.append('num_registros')
        agregacao.columns = colunas_base
        
        if 'risco_fogo' not in agregacao.columns:
            agregacao['risco_fogo'] = 0
        
        return agregacao.reset_index(drop=True)
    
    def __construir_arestas_agregadas(
        self, 
        vertices: List[int], 
        threshold_km: float, 
        threshold_dias: int,
        usar_temporal: bool, 
        usar_espacial: bool,
        max_conexoes: int, 
        mostrar_progresso: bool
    ) -> None:
        
        total_vertices = len(vertices)
        
        for i, v1 in enumerate(vertices):
            if mostrar_progresso and i % 500 == 0:
                print(f'Conectando vértice {i+1}/{total_vertices}')
                
            dados_v1 = self.__vertice_data[v1]
            conexoes_v1 = 0
            
            for v2 in vertices[i+1:]:
                if conexoes_v1 >= max_conexoes:
                    break
                    
                dados_v2 = self.__vertice_data[v2]
                
                conectar, peso = self.__verificar_conexao(
                    dados_v1, dados_v2, threshold_km, threshold_dias,
                    usar_temporal, usar_espacial
                )
                
                if conectar:
                    self.add_ponto_con(v1, v2, peso)
                    conexoes_v1 += 1
    
    def __verificar_conexao(
        self, 
        dados_v1: Dict, 
        dados_v2: Dict,
        threshold_km: float, 
        threshold_dias: int,
        usar_temporal: bool, 
        usar_espacial: bool
    ) -> Tuple[bool, float]:
        
        conectar = False
        peso_total = 0.0
        componentes_peso = 0
        
        if usar_espacial:
            dist_km = self.calcular_distancia_haversine(
                dados_v1['latitude'], dados_v1['longitude'],
                dados_v2['latitude'], dados_v2['longitude']
            )
            
            if dist_km <= threshold_km:
                conectar = True
                peso_total += dist_km / threshold_km
                componentes_peso += 1
                
        if usar_temporal and dados_v1['data'] is not pd.NaT and dados_v2['data'] is not pd.NaT:
            diff_dias = self.calcular_diferenca_dias(dados_v1['data'], dados_v2['data'])
            
            if diff_dias <= threshold_dias:
                conectar = True
                peso_total += diff_dias / threshold_dias
                componentes_peso += 1
        
        peso_medio = peso_total / componentes_peso if componentes_peso > 0 else 0.0
        return conectar, peso_medio
        
    def get_vizinhos(self, vertice_id: int) -> List[int]:
        return super().get_vizinhos(vertice_id)
    
    def get_peso(self, v1: int, v2: int) -> float:
        return super().get_peso(v1, v2)
        
    def calcular_grau(self, vertice_id: int) -> int:
        return len(self.get_vizinhos(vertice_id))
    
    def calcular_peso_medio_arestas(self, vertice_id: int) -> float:
        vizinhos = self.get_vizinhos(vertice_id)
        if not vizinhos:
            return 0.0
        pesos = [self.get_peso(vertice_id, v) for v in vizinhos]
        return np.mean(pesos)
    
    def calcular_risco_medio_vizinhos(self, vertice_id: int) -> float:
        vizinhos = self.get_vizinhos(vertice_id)
        if not vizinhos:
            return 0.0
        riscos = [self.__vertice_data[v]['risco_fogo'] for v in vizinhos if v in self.__vertice_data]
        return np.mean(riscos) if riscos else 0.0
    
    def calcular_coeficiente_clustering(self, vertice_id: int) -> float:
        vizinhos = self.get_vizinhos(vertice_id)
        k = len(vizinhos)
        if k < 2:
            return 0.0
        arestas_entre_vizinhos = 0
        for i, v1 in enumerate(vizinhos):
            for v2 in vizinhos[i+1:]:
                if self.get_peso(v1, v2) > 0:
                    arestas_entre_vizinhos += 1
        arestas_possiveis = k * (k - 1) / 2
        return arestas_entre_vizinhos / arestas_possiveis if arestas_possiveis > 0 else 0.0
    
    def calcular_centralidade_grau(self, vertice_id: int) -> float:
        n = len(self.get_pontos())
        if n <= 1:
            return 0.0
        grau = self.calcular_grau(vertice_id)
        return grau / (n - 1)
    
    def calcular_risco_propagacao(self, vertice_id: int) -> float:
        grau = self.calcular_grau(vertice_id)
        risco_medio_viz = self.calcular_risco_medio_vizinhos(vertice_id)
        dados = self.__vertice_data.get(vertice_id, {})
        risco_local = dados.get('risco_fogo', 0)
        return (risco_local * 0.5 + risco_medio_viz * 0.3 + grau * 0.2)
    
    def extrair_features_vertice(self, vertice_id: int) -> Dict[str, float]:
        if vertice_id in self.__features_cache:
            return self.__features_cache[vertice_id].copy()
        
        features = {
            'grau': float(self.calcular_grau(vertice_id)),
            'peso_medio_arestas': self.calcular_peso_medio_arestas(vertice_id),
            'risco_medio_vizinhos': self.calcular_risco_medio_vizinhos(vertice_id),
            'coeficiente_clustering': self.calcular_coeficiente_clustering(vertice_id),
            'centralidade_grau': self.calcular_centralidade_grau(vertice_id),
            'risco_propagacao': self.calcular_risco_propagacao(vertice_id)
        }
        
        self.__features_cache[vertice_id] = features
        return features.copy()
    
    def extrair_features_dataframe(self, df_original: pd.DataFrame) -> pd.DataFrame:
        df = df_original.copy()
        vertices = self.get_pontos()
        
        print(f'Extraindo features de {len(vertices)} vértices...')
        
        graus = []
        pesos_medios = []
        riscos_vizinhos = []
        coefs_clustering = []
        centralidades = []
        riscos_propagacao = []
        
        for i, v_id in enumerate(vertices):
            if i % 1000 == 0 and i > 0:
                print(f'  Processado {i}/{len(vertices)} vértices')
            
            vizinhos = self.get_vizinhos(v_id)
            grau = len(vizinhos)
            graus.append(grau)
            
            if vizinhos:
                pesos = [self.get_peso(v_id, v) for v in vizinhos]
                pesos_medios.append(np.mean(pesos))
                riscos = [self.__vertice_data[v]['risco_fogo'] for v in vizinhos if v in self.__vertice_data]
                riscos_vizinhos.append(np.mean(riscos) if riscos else 0.0)
            else:
                pesos_medios.append(0.0)
                riscos_vizinhos.append(0.0)
            
            if grau < 2:
                coefs_clustering.append(0.0)
            else:
                arestas = sum(1 for i, v1 in enumerate(vizinhos) for v2 in vizinhos[i+1:] if self.get_peso(v1, v2) > 0)
                coefs_clustering.append(arestas / (grau * (grau - 1) / 2) if grau > 1 else 0.0)
            
            n = len(vertices)
            centralidades.append(grau / (n - 1) if n > 1 else 0.0)
            
            risco_local = self.__vertice_data[v_id]['risco_fogo']
            riscos_propagacao.append(risco_local * 0.5 + riscos_vizinhos[-1] * 0.3 + grau * 0.2)
        
        df_features = pd.DataFrame({
            'grafo_grau': graus,
            'grafo_peso_medio_arestas': pesos_medios,
            'grafo_risco_medio_vizinhos': riscos_vizinhos,
            'grafo_coeficiente_clustering': coefs_clustering,
            'grafo_centralidade_grau': centralidades,
            'grafo_risco_propagacao': riscos_propagacao
        })
        
        if len(df_features) < len(df):
            df_features = df_features.reindex(range(len(df)), fill_value=0)
        elif len(df_features) > len(df):
            df_features = df_features.iloc[:len(df)]
        
        for col in df_features.columns:
            df[col] = df_features[col].values
        
        print(f'Features extraídas com sucesso!')
        return df
    
    def identificar_regioes_criticas(self, percentil: float = 90) -> List[Tuple[int, float, float]]:
        vertices = self.get_pontos()
        if not vertices:
            return []
        
        centralidades = [(v, self.calcular_centralidade_grau(v), 
                         self.__vertice_data[v]['risco_fogo']) 
                        for v in vertices]
        
        threshold = np.percentile([c[1] for c in centralidades], percentil)
        regioes_criticas = [c for c in centralidades if c[1] >= threshold]
        regioes_criticas.sort(key=lambda x: x[1], reverse=True)
        
        return regioes_criticas
