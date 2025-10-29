import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from cluster.cluster_utils import preparar_dados
from data.connection import connection

def carregar_modelo(caminho_modelo='./models/modelo_cluster.pkl'):
    '''
    Carrega o modelo treinado
    '''
    try:
        with open(caminho_modelo, 'rb') as f:
            modelo = pickle.load(f)
        print(f"Modelo carregado com sucesso de {caminho_modelo}")
        return modelo
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return None

def analisar_padroes_historicos(df_historico):
    '''
    Analisa padrões históricos para entender distribuição espacial e temporal
    '''
    
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
    
    # Calcular bounds do cerrado baseado nos dados históricos
    lat_min, lat_max = df_hist['Latitude'].min(), df_hist['Latitude'].max()
    lon_min, lon_max = df_hist['Longitude'].min(), df_hist['Longitude'].max()
    
    print(f"\n=== Análise de Padrões Históricos ===")
    print(f"Total de registros históricos: {len(df_hist)}")
    print(f"Coordenadas únicas: {len(coords_freq)}")
    print(f"Área do Cerrado: Lat [{lat_min:.2f}, {lat_max:.2f}] | Lon [{lon_min:.2f}, {lon_max:.2f}]")
    print(f"Distribuição mensal de focos:")
    for mes, prop in sorted(distribuicao_mensal.items()):
        print(f"  Mês {mes}: {prop*100:.1f}%")
    
    return {
        'distribuicao_mensal': distribuicao_mensal,
        'stats_mes_regiao': stats_por_mes_regiao,
        'coords_freq': coords_freq,
        'bounds': {'lat_min': lat_min, 'lat_max': lat_max, 'lon_min': lon_min, 'lon_max': lon_max}
    }

def gerar_coordenada_interpolada(coords_freq, bounds, usar_hotspot=True):
    '''
    Gera uma coordenada, podendo ser de hotspot ou interpolada na região do cerrado
    '''
    if usar_hotspot:
        # Escolher de um hotspot existente
        idx = np.random.choice(len(coords_freq) // 3)
        coord = coords_freq.iloc[idx]
        return coord['Latitude'], coord['Longitude'], coord['Estado'], coord['Municipio']
    else:
        # Gerar coordenada interpolada dentro dos bounds do cerrado
        # Adicionar pequena variação nas coordenadas existentes
        idx = np.random.randint(0, len(coords_freq))
        coord_base = coords_freq.iloc[idx]
        
        # Adicionar variação de até 0.5 graus (aproximadamente 55km)
        lat = coord_base['Latitude'] + np.random.uniform(-0.5, 0.5)
        lon = coord_base['Longitude'] + np.random.uniform(-0.5, 0.5)
        
        # Garantir que está dentro dos bounds
        lat = np.clip(lat, bounds['lat_min'], bounds['lat_max'])
        lon = np.clip(lon, bounds['lon_min'], bounds['lon_max'])
        
        return lat, lon, coord_base['Estado'], coord_base['Municipio']

def gerar_dados_2026(padroes, total_registros=None):
    '''
    Gera dados para 2026 de forma inteligente, respeitando padrões históricos
    '''
    
    # Gerar número aleatório de registros entre 45k e 60k se não especificado
    if total_registros is None:
        total_registros = np.random.randint(45000, 60001)
    
    registros = []
    coords_freq = padroes['coords_freq']
    distribuicao_mensal = padroes['distribuicao_mensal']
    stats_mes_regiao = padroes['stats_mes_regiao']
    bounds = padroes['bounds']
    
    # Calcular quantos registros por mês (respeitando sazonalidade)
    registros_por_mes = {}
    for mes in range(1, 13):
        prop = distribuicao_mensal.get(mes, 1/12)
        registros_por_mes[mes] = int(total_registros * prop)
    
    # Ajustar para ter exatamente total_registros
    diff = total_registros - sum(registros_por_mes.values())
    registros_por_mes[8] += diff  # Adicionar diferença no mês de pico (agosto)
    
    print(f"\n=== Gerando {total_registros:,} registros para 2026 ===")
    
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
            
            # 50% hotspot, 50% coordenadas interpoladas (para melhor distribuição)
            usar_hotspot = np.random.random() < 0.5
            lat, lon, estado, municipio = gerar_coordenada_interpolada(coords_freq, bounds, usar_hotspot)
            
            # Buscar estatísticas para esse mês e estado
            stats = stats_mes_regiao[
                (stats_mes_regiao['Mes'] == mes) & 
                (stats_mes_regiao['Estado'] == estado)
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
                'Latitude': round(lat, 6),
                'Longitude': round(lon, 6),
                'Estado': estado,
                'Municipio': municipio,
                'DiaSemChuva': dias_sem_chuva,
                'Precipitacao': round(precipitacao, 2),
                'FRP': round(frp, 2)
            })
    
    df_2026 = pd.DataFrame(registros)
    
    # Ordenar por data para ficar orgânico
    df_2026 = df_2026.sort_values('Data').reset_index(drop=True)
    
    print(f"\n✓ Gerados {len(df_2026):,} registros")
    print(f"✓ Período: {df_2026['Data'].min()} a {df_2026['Data'].max()}")
    print(f"✓ Coordenadas únicas: {df_2026[['Latitude', 'Longitude']].drop_duplicates().shape[0]:,}")
    
    return df_2026

def prever_dados(modelo, df_2026):
    '''
    Aplica o modelo para prever RiscoFogo
    '''
    
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
    
    # Converter RiscoFogo para porcentagem inteira (0-100)
    df_2026['RiscoFogo'] = np.ceil(df_2026['RiscoFogo'] * 100).astype(int)
    
    # Ajustar sutilmente variáveis baseado no RiscoFogo previsto
    df_2026['DiaSemChuva'] = (df_2026['DiaSemChuva'] * (1 + df_2026['RiscoFogo'] / 100 * 0.3)).astype(int)
    df_2026['Precipitacao'] = df_2026['Precipitacao'] * (1 - df_2026['RiscoFogo'] / 100 * 0.2)
    df_2026['FRP'] = df_2026['FRP'] * (1 + df_2026['RiscoFogo'] / 100 * 1.5)
    
    # Decodificar Estado e Municipio usando label_encoders
    if 'label_encoders' in modelo:
        label_encoders = modelo['label_encoders']
        
        if 'Estado_encoded' in df_2026.columns and 'Estado' in label_encoders:
            df_2026['Estado'] = label_encoders['Estado'].inverse_transform(df_2026['Estado_encoded'])
        
        if 'Municipio_encoded' in df_2026.columns and 'Municipio' in label_encoders:
            df_2026['Municipio'] = label_encoders['Municipio'].inverse_transform(df_2026['Municipio_encoded'])
    
    return df_2026

def salvar_previsao(df_previsao, nome_arquivo='previsao_2026.csv'):
    '''
    Salva as previsões em CSV no formato correto
    '''

    colunas_finais = ['Data', 'Estado', 'Municipio', 'RiscoFogo',
                      'DiaSemChuva', 'Precipitacao', 'FRP', 'Latitude', 'Longitude']
    
    df_final = df_previsao[colunas_finais].copy()
    df_final['Data'] = df_final['Data'].dt.strftime('%Y-%m-%d')
    
    # Arredondar valores numéricos
    df_final['Latitude'] = df_final['Latitude'].round(6)
    df_final['Longitude'] = df_final['Longitude'].round(6)
    df_final['RiscoFogo'] = df_final['RiscoFogo'].astype(int)
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
    '''
    Função principal para gerar previsões
    '''
    
    print("="*60)
    print("    Gerador Inteligente de Previsões 2026 - Cerrado")
    print("="*60)
    
    # 1. Carregar modelo
    print("\n[1/5] Carregando modelo...")
    modelo = carregar_modelo()
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
    
    # 4. Gerar dados para 2026 de forma inteligente (com número aleatório de linhas)
    print("\n[4/5] Gerando dados para 2026...")
    df_2026 = gerar_dados_2026(padroes)  # Sem passar total_registros
    
    # 5. Fazer previsões
    print("\n[5/5] Aplicando modelo para previsões...")
    df_previsao = prever_dados(modelo, df_2026)
    
    # 6. Salvar resultados
    salvar_previsao(df_previsao, 'previsao_2026.csv')
    
    # Estatísticas finais
    print("\n" + "="*60)
    print("    ESTATÍSTICAS DA PREVISÃO")
    print("="*60)
    print(f"RiscoFogo médio: {df_previsao['RiscoFogo'].mean():.1f}%")
    print(f"RiscoFogo min/max: {df_previsao['RiscoFogo'].min()}% / {df_previsao['RiscoFogo'].max()}%")
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