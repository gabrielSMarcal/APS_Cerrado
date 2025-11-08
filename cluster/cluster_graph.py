import numpy as np
import pandas as pd
from typing import Tuple

from models.TAD.ClusterGraph import ClusterGraph


def construir_grafo_espacial(df: pd.DataFrame, threshold_km: float = 50.0, grid_size_km: float = 50.0) -> ClusterGraph:
    grafo = ClusterGraph()
    grafo.construir_grafo_dataframe(
        df,
        threshold_km=threshold_km,
        threshold_dias=0,
        usar_temporal=False,
        usar_espacial=True,
        max_conexoes_por_vertice=10,
        grid_size_km=grid_size_km,
        janela_temporal_dias=30
    )
    return grafo


def construir_grafo_temporal(df: pd.DataFrame, threshold_dias: int = 7, janela_temporal_dias: int = 14) -> ClusterGraph:
    grafo = ClusterGraph()
    grafo.construir_grafo_dataframe(
        df,
        threshold_km=0,
        threshold_dias=threshold_dias,
        usar_temporal=True,
        usar_espacial=False,
        max_conexoes_por_vertice=10,
        grid_size_km=100.0,
        janela_temporal_dias=janela_temporal_dias
    )
    return grafo


def construir_grafo_hibrido(
    df: pd.DataFrame, 
    threshold_km: float = 50.0, 
    threshold_dias: int = 7,
    grid_size_km: float = 50.0,
    janela_temporal_dias: int = 14,
    max_conexoes_por_vertice: int = 10
) -> ClusterGraph:
    grafo = ClusterGraph()
    grafo.construir_grafo_dataframe(
        df,
        threshold_km=threshold_km,
        threshold_dias=threshold_dias,
        usar_temporal=True,
        usar_espacial=True,
        max_conexoes_por_vertice=max_conexoes_por_vertice,
        grid_size_km=grid_size_km,
        janela_temporal_dias=janela_temporal_dias
    )
    return grafo


def extrair_features_grafo(grafo: ClusterGraph, df: pd.DataFrame) -> pd.DataFrame:
    df_features = grafo.extrair_features_dataframe(df)
    return df_features


def analisar_regioes_criticas(grafo: ClusterGraph, df: pd.DataFrame, percentil: float = 90) -> pd.DataFrame:
    regioes = grafo.identificar_regioes_criticas(percentil)
    
    if not regioes:
        return pd.DataFrame()
    
    dados_regioes = []
    for vertice_id, centralidade, risco in regioes:
        dados_vertice = grafo.get_dados_vertice(vertice_id)
        dados_regioes.append({
            'vertice_id': vertice_id,
            'estado': dados_vertice.get('estado', ''),
            'municipio': dados_vertice.get('municipio', ''),
            'latitude': dados_vertice.get('latitude', 0.0),
            'longitude': dados_vertice.get('longitude', 0.0),
            'risco_fogo': risco,
            'centralidade': centralidade,
            'grau': grafo.calcular_grau(vertice_id),
            'risco_propagacao': grafo.calcular_risco_propagacao(vertice_id)
        })
    
    df_regioes = pd.DataFrame(dados_regioes)
    return df_regioes


def preparar_dados_com_grafo(
    df: pd.DataFrame,
    threshold_km: float = 50.0,
    threshold_dias: int = 7,
    tipo_grafo: str = 'hibrido',
    grid_size_km: float = 50.0,
    janela_temporal_dias: int = 14,
    max_conexoes_por_vertice: int = 10
) -> Tuple[pd.DataFrame, ClusterGraph]:

    print(f"\n{'='*60}")
    print(f"Preparando dados com grafo tipo: {tipo_grafo}")
    print(f"{'='*60}")

    if tipo_grafo == 'espacial':
        grafo = construir_grafo_espacial(df, threshold_km, grid_size_km)
    elif tipo_grafo == 'temporal':
        grafo = construir_grafo_temporal(df, threshold_dias, janela_temporal_dias)
    else:
        grafo = construir_grafo_hibrido(df, threshold_km, threshold_dias, grid_size_km, janela_temporal_dias, max_conexoes_por_vertice)
    
    df_com_features = grafo.extrair_features_dataframe(df)

    print(f"\nFeatures do grafo adicionadas ao DataFrame")
    print(f"Shape do DataFrame: {df_com_features.shape}")

    return df_com_features, grafo


def calcular_estatisticas_grafo(grafo: ClusterGraph) -> dict:
    vertices = grafo.get_pontos()
    
    if not vertices:
        return {}
    
    graus = [grafo.calcular_grau(v) for v in vertices]
    centralidades = [grafo.calcular_centralidade_grau(v) for v in vertices]
    coefs_clustering = [grafo.calcular_coeficiente_clustering(v) for v in vertices]
    
    estatisticas = {
        'num_vertices': len(vertices),
        'num_arestas': grafo.total_ponto_con(),
        'grau_medio': np.mean(graus),
        'grau_max': np.max(graus),
        'grau_min': np.min(graus),
        'centralidade_media': np.mean(centralidades),
        'coef_clustering_medio': np.mean(coefs_clustering),
        'densidade': grafo.total_ponto_con() / (len(vertices) * (len(vertices) - 1) / 2) if len(vertices) > 1 else 0
    }
    
    return estatisticas


def imprimir_estatisticas_grafo(grafo: ClusterGraph) -> None:
    stats = calcular_estatisticas_grafo(grafo)
    
    if not stats:
        print("Grafo vazio, sem estatísticas para exibir.")
        return

    print(f"\n{'='*60}")
    print(f"ESTATÍSTICAS DO GRAFO")
    print(f"{'='*60}")
    print(f"Número de vértices: {stats['num_vertices']}")
    print(f"Número de arestas: {stats['num_arestas']}")
    print(f"Densidade do grafo: {stats['densidade']:.4f}")
    print(f"\nGrau dos vértices:")
    print(f"  - Médio: {stats['grau_medio']:.2f}")
    print(f"  - Mínimo: {stats['grau_min']}")
    print(f"  - Máximo: {stats['grau_max']}")
    print(f"\nCentralidade média: {stats['centralidade_media']:.4f}")
    print(f"Coeficiente de clustering médio: {stats['coef_clustering_medio']:.4f}")
    print(f"{'='*60}\n")


def otimizar_thresholds(
    df: pd.DataFrame,
    threshold_km_range: list = [30, 50, 100],
    threshold_dias_range: list = [3, 7, 14],
    grid_size_km: float = 50.0,
    janela_temporal_dias: int = 14,
    max_conexoes_por_vertice: int = 10
) -> Tuple[float, int, dict]:

    print(f"\n{'='*60}")
    print(f"OTIMIZANDO THRESHOLDS DO GRAFO")
    print(f"{'='*60}")

    resultados = []
    
    for threshold_km in threshold_km_range:
        for threshold_dias in threshold_dias_range:
            print(f"\nTestando: threshold_km={threshold_km}, threshold_dias={threshold_dias}")
            
            grafo = construir_grafo_hibrido(df, threshold_km, threshold_dias, grid_size_km, janela_temporal_dias, max_conexoes_por_vertice)
            stats = calcular_estatisticas_grafo(grafo)
            
            if stats:
                score = stats['grau_medio'] * (1 - abs(stats['densidade'] - 0.1))
                
                resultados.append({
                    'threshold_km': threshold_km,
                    'threshold_dias': threshold_dias,
                    'score': score,
                    'stats': stats
                })
                
                print(f'  Score: {score:.4f} | Grau médio: {stats["grau_medio"]:.2f} | '
                      f'Densidade: {stats["densidade"]:.4f}')

    if not resultados:
        print('Nenhum resultado válido encontrado.')
        return 50.0, 7, {}
    
    resultados.sort(key=lambda x: x['score'], reverse=True)
    melhor = resultados[0]

    print(f"\n{'='*60}")
    print(f"MELHOR CONFIGURAÇÃO ENCONTRADA")
    print(f"{'='*60}")
    print(f"Threshold KM: {melhor['threshold_km']}")
    print(f"Threshold Dias: {melhor['threshold_dias']}")
    print(f"Score: {melhor['score']:.4f}")
    print(f"{'='*60}\n")

    return melhor['threshold_km'], melhor['threshold_dias'], melhor['stats']
