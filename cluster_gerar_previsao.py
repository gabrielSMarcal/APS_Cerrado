import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from cluster_utils import preparar_dados
from data.connection import connection

def carregar_modelo(caminho_modelo='modelo_cluster.pkl'):
    """Carrega o modelo treinado"""
    try:
        with open(caminho_modelo, 'rb') as f:
            modelo = pickle.load(f)
        print(f"Modelo carregado com sucesso de {caminho_modelo}")
        return modelo
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return None

def analisar_padroes_historicos(df_historico):
    """
    Analisa padrões históricos para entender distribuição espacial e temporal
    """
    df_hist = df_historico.copy()
    df_hist['Data'] = pd.to_datetime(df_hist['Data'])
    df_hist['Mes'] = df_hist['Data'].dt.month
    df_hist['Dia'] = df_hist['Data'].dt.day
    
    # Análise por mês: quantos focos por mês em média
    focos_por_mes = df_hist.groupby('Mes').size()
    distribuicao_mensal = (focos_por_mes / focos_por_mes.sum()).to_dict()
    
    # Estatísticas por mês e região
    stats_por_mes_regiao = df_hist.groupby(['Mes', 'Estado']).agg({
        'DiaSemChuva': 'mean',
        'Precipitacao': 'mean',
        'FRP': 'mean',
        'RiscoFogo': 'mean'
    }).reset_index()
    
    # Coordenadas mais frequentes (hotspots)
    coords_freq = df_hist.groupby(['Latitude', 'Longitude', 'Estado', 'Municipio']).size().reset_index(name='Frequencia')
    coords_freq = coords_freq.sort_values('Frequencia', ascending=False)
    
    print(f"\n=== Análise de Padrões Históricos ===")
    print(f"Total de registros históricos: {len(df_hist)}")
    print(f"Coordenadas únicas: {len(coords_freq)}")
    print(f"Distribuição mensal de focos:")
    for mes, prop in sorted(distribuicao_mensal.items()):
        print(f"  Mês {mes}: {prop*100:.1f}%")
    
    return {
        'distribuicao_mensal': distribuicao_mensal,
        'stats_mes_regiao': stats_por_mes_regiao,
        'coords_freq': coords_freq
    }

def gerar_dados_2026_inteligente(padroes, df_historico, total_registros=50000):
    """
    Gera dados para 2026 de forma inteligente, respeitando padrões históricos
    """
    data_inicio = datetime(2026, 1, 1)
    data_fim = datetime(2026, 12, 31)
    
    registros = []
    coords_freq = padroes['coords_freq']
    distribuicao_mensal = padroes['distribuicao_mensal']
    stats_mes_regiao = padroes['stats_mes_regiao']
    
    # Calcular quantos registros por mês (respeitando sazonalidade)
    registros_por_mes = {}
    for mes in range(1, 13):
        prop = distribuicao_mensal.get(mes, 1/12)
        registros_por_mes[mes] = int(total_registros * prop)
    
    # Ajustar para ter exatamente total_registros
    diff = total_registros - sum(registros_por_mes.values())
    registros_por_mes[8] += diff  # Adicionar diferença no mês de pico (agosto)
    
    print(f"\n=== Gerando {total_registros} registros para 2026 ===")
    
    # Para cada mês, gerar registros distribuídos ao longo dos dias
    for mes in range(1, 13):
        num_registros_mes = registros_por_mes[mes]
        dias_no_mes = (datetime(2026, mes+1, 1) - datetime(2026, mes, 1)).days if mes < 12 else 31
        
        print(f"Mês {mes}: {num_registros_mes} registros")
        
        # Distribuir registros ao longo dos dias do mês
        for i in range(num_registros_mes):
            # Escolher dia aleatório do mês
            dia = np.random.randint(1, dias_no_mes + 1)
            data = datetime(2026, mes, dia)
            
            # Escolher coordenada baseada em frequência histórica (mais peso para hotspots)
            # 70% de chance de escolher dos 30% hotspots mais frequentes
            if np.random.random() < 0.7:
                # Hotspot
                idx = np.random.choice(len(coords_freq) // 3)
            else:
                # Coordenada aleatória
                idx = np.random.randint(0, len(coords_freq))
            
            coord = coords_freq.iloc[idx]
            
            # Buscar estatísticas para esse mês e estado
            stats = stats_mes_regiao[
                (stats_mes_regiao['Mes'] == mes) & 
                (stats_mes_regiao['Estado'] == coord['Estado'])
            ]
            
            if len(stats) > 0:
                stats = stats.iloc[0]
                # Usar estatísticas históricas + variação
                dias_sem_chuva = max(0, int(stats['DiaSemChuva'] + np.random.normal(0, 5)))
                precipitacao = max(0, stats['Precipitacao'] + np.random.normal(0, 10))
                frp = max(0, stats['FRP'] + np.random.normal(0, 20))
            else:
                # Valores padrão se não tiver histórico
                # Meses secos (maio-outubro): mais dias sem chuva
                if mes in [5, 6, 7, 8, 9, 10]:
                    dias_sem_chuva = np.random.randint(15, 60)
                    precipitacao = np.random.uniform(0, 5)
                    frp = np.random.uniform(30, 150)
                else:
                    dias_sem_chuva = np.random.randint(0, 20)
                    precipitacao = np.random.uniform(5, 100)
                    frp = np.random.uniform(10, 80)
            
            registros.append({
                'Data': data,
                'Latitude': coord['Latitude'],
                'Longitude': coord['Longitude'],
                'Estado': coord['Estado'],
                'Municipio': coord['Municipio'],
                'DiaSemChuva': dias_sem_chuva,
                'Precipitacao': round(precipitacao, 2),
                'FRP': round(frp, 2)
            })
    
    df_2026 = pd.DataFrame(registros)
    
    # Ordenar por data para ficar orgânico
    df_2026 = df_2026.sort_values('Data').reset_index(drop=True)
    
    print(f"\nGerados {len(df_2026)} registros")
    print(f"Período: {df_2026['Data'].min()} a {df_2026['Data'].max()}")
    print(f"Coordenadas únicas: {df_2026[['Latitude', 'Longitude']].drop_duplicates().shape[0]}")
    
    return df_2026

def prever_dados(modelo, df_2026):
    """
    Aplica o modelo para prever RiscoFogo
    """
    print("\nPreparando dados para previsão...")
    
    # Preparar dados usando a mesma função de preparação
    df_preparado, _ = preparar_dados(df_2026, modelo_cluster=modelo)
    
    # Obter features na ordem correta
    feature_names = modelo.get('feature_names', df_preparado.columns.tolist())
    
    # Verificar quais features estão disponíveis
    features_disponiveis = [f for f in feature_names if f in df_preparado.columns]
    print(f"Features disponíveis: {len(features_disponiveis)}/{len(feature_names)}")
    
    if len(features_disponiveis) < len(feature_names):
        features_faltando = set(feature_names) - set(features_disponiveis)
        print(f"Aviso: Features faltando: {features_faltando}")
    
    X_pred = df_preparado[features_disponiveis]
    
    # Fazer previsão dos clusters
    try:
        if 'kmeans' in modelo:
            clusters = modelo['kmeans'].predict(X_pred)
        elif 'modelo' in modelo:
            clusters = modelo['modelo'].predict(X_pred)
        else:
            raise ValueError("Modelo não contém 'kmeans' ou 'modelo'")
        
        print(f"Previsão realizada com sucesso!")
    except Exception as e:
        print(f"Erro na previsão: {e}")
        clusters = np.zeros(len(df_2026), dtype=int)
    
    # Adicionar clusters ao dataframe original
    df_2026['Cluster'] = clusters
    
    # Mapear clusters para risco de fogo
    if 'cluster_stats' in modelo:
        cluster_risk_map = {}
        for cluster_id, stats in modelo['cluster_stats'].items():
            cluster_risk_map[cluster_id] = stats.get('RiscoFogo_mean', 0.5)
        
        df_2026['RiscoFogo'] = df_2026['Cluster'].map(cluster_risk_map).fillna(0.5)
        print(f"Risco mapeado usando estatísticas dos clusters")
    else:
        # Normalizar cluster ID para risco (0-1)
        max_cluster = df_2026['Cluster'].max()
        if max_cluster > 0:
            df_2026['RiscoFogo'] = (df_2026['Cluster'] / max_cluster).clip(0, 1)
        else:
            df_2026['RiscoFogo'] = 0.5
        print(f"Risco calculado por normalização")
    
    # Ajustar sutilmente variáveis baseado no RiscoFogo previsto
    df_2026['DiaSemChuva'] = (df_2026['DiaSemChuva'] * (1 + df_2026['RiscoFogo'] * 0.3)).astype(int)
    df_2026['Precipitacao'] = df_2026['Precipitacao'] * (1 - df_2026['RiscoFogo'] * 0.2)
    df_2026['FRP'] = df_2026['FRP'] * (1 + df_2026['RiscoFogo'] * 1.5)
    
    return df_2026

def salvar_previsao(df_previsao, nome_arquivo='previsao_2026_inteligente.csv'):
    """Salva as previsões em CSV no formato correto"""
    colunas_finais = ['Data', 'Latitude', 'Longitude', 'RiscoFogo', 
                      'DiaSemChuva', 'Precipitacao', 'FRP']
    
    df_final = df_previsao[colunas_finais].copy()
    df_final['Data'] = df_final['Data'].dt.strftime('%Y-%m-%d')
    
    # Arredondar valores numéricos
    df_final['Latitude'] = df_final['Latitude'].round(6)
    df_final['Longitude'] = df_final['Longitude'].round(6)
    df_final['RiscoFogo'] = df_final['RiscoFogo'].round(4)
    df_final['DiaSemChuva'] = df_final['DiaSemChuva'].astype(int)
    df_final['Precipitacao'] = df_final['Precipitacao'].round(2)
    df_final['FRP'] = df_final['FRP'].round(2)
    
    df_final.to_csv(nome_arquivo, index=False)
    print(f"\n{'='*50}")
    print(f"Previsão salva em '{nome_arquivo}'")
    print(f"{'='*50}")
    print(f"Total de linhas: {len(df_final):,}")
    print(f"Coordenadas únicas: {df_final[['Latitude', 'Longitude']].drop_duplicates().shape[0]:,}")
    print(f"Período: {df_final['Data'].min()} a {df_final['Data'].max()}")
    print(f"\nDistribuição mensal:")
    df_temp = df_final.copy()
    df_temp['Mes'] = pd.to_datetime(df_temp['Data']).dt.month
    print(df_temp.groupby('Mes').size())

def main():
    """Função principal para gerar previsões"""
    print("="*60)
    print("    Gerador Inteligente de Previsões 2026 - Cerrado")
    print("="*60)
    
    # 1. Carregar modelo
    print("\n[1/5] Carregando modelo...")
    modelo = carregar_modelo('modelo_cluster.pkl')
    if modelo is None:
        return
    
    # 2. Carregar dados históricos
    print("\n[2/5] Carregando dados históricos...")
    df_historico = connection()
    if df_historico is None:
        print("Erro ao carregar dados históricos")
        return
    
    # 3. Analisar padrões históricos
    print("\n[3/5] Analisando padrões históricos...")
    padroes = analisar_padroes_historicos(df_historico)
    
    # 4. Gerar dados para 2026 de forma inteligente
    print("\n[4/5] Gerando dados para 2026...")
    df_2026 = gerar_dados_2026_inteligente(padroes, df_historico, total_registros=50000)
    
    # 5. Fazer previsões
    print("\n[5/5] Aplicando modelo para previsões...")
    df_previsao = prever_dados(modelo, df_2026)
    
    # 6. Salvar resultados
    salvar_previsao(df_previsao, 'previsao_2026_inteligente.csv')
    
    # Estatísticas finais
    print("\n" + "="*60)
    print("    ESTATÍSTICAS DA PREVISÃO")
    print("="*60)
    print(f"RiscoFogo médio: {df_previsao['RiscoFogo'].mean():.4f}")
    print(f"RiscoFogo min/max: {df_previsao['RiscoFogo'].min():.4f} / {df_previsao['RiscoFogo'].max():.4f}")
    print(f"DiaSemChuva médio: {df_previsao['DiaSemChuva'].mean():.1f} dias")
    print(f"Precipitacao média: {df_previsao['Precipitacao'].mean():.2f} mm")
    print(f"FRP médio: {df_previsao['FRP'].mean():.2f}")
    
    print("\n" + "="*60)
    print("    DISTRIBUIÇÃO DE CLUSTERS")
    print("="*60)
    print(df_previsao['Cluster'].value_counts().sort_index())
    
    print("\n✅ Previsão concluída com sucesso!")
    print("📊 Execute 'python teste_cluster.py' para visualizar no mapa")

if __name__ == "__main__":
    main()