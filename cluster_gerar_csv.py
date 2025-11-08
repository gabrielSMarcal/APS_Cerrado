import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime
from cluster.preparacao_dados import preparar_para_predicao, criar_variaveis_temporais
from data.connection import connection
from models.TAD.ClusterGraph import ClusterGraph

def carregar_modelo(caminho_modelo='./source/modelo_random_forest.pkl'):
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
    
    print(f'\n=== Análise de Padrões Históricos ===')
    print(f'Total de registros históricos: {len(df_hist)}')
    print(f'Coordenadas únicas: {len(coords_freq)}')
    print(f'Área do Cerrado: Lat [{lat_min:.2f}, {lat_max:.2f}] | Lon [{lon_min:.2f}, {lon_max:.2f}]')
    print(f'Distribuição mensal de focos:')
    for mes, prop in sorted(distribuicao_mensal.items()):
        print(f'  Mês {mes}: {prop*100:.1f}%')
    
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
    
    print(f'\n=== Gerando {total_registros:,} registros para 2026 ===')
    
    # Para cada mês, gerar registros distribuídos ao longo dos dias
    for mes in range(1, 13):
        num_registros_mes = registros_por_mes[mes]
        dias_no_mes = (datetime(2026, mes+1, 1) - datetime(2026, mes, 1)).days if mes < 12 else 31
        
        print(f'Mês {mes}: {num_registros_mes} registros')
        
        # Distribuir registros ao longo dos dias do mês
        for i in range(num_registros_mes):
            dia = np.random.randint(1, dias_no_mes + 1)
            data = datetime(2026, mes, dia)
            
            # 50% hotspot, 50% coordenadas interpoladas
            usar_hotspot = np.random.random() < 0.5
            lat, lon, estado, municipio = gerar_coordenada_interpolada(coords_freq, bounds, usar_hotspot)
            
            # Buscar estatísticas para esse mês e estado
            stats = stats_mes_regiao[
                (stats_mes_regiao['Mes'] == mes) & 
                (stats_mes_regiao['Estado'] == estado)
            ]
            
            if len(stats) > 0:
                stats = stats.iloc[0]
                dias_sem_chuva = max(0, int(stats['DiaSemChuva'] + np.random.normal(0, 5)))
                precipitacao = max(0, stats['Precipitacao'] + np.random.normal(0, 10))
                frp = max(0, stats['FRP'] + np.random.normal(0, 20))
            else:
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
    df_2026 = df_2026.sort_values('Data').reset_index(drop=True)

    print(f'\n✓ Gerados {len(df_2026):,} registros')
    print(f'✓ Período: {df_2026["Data"].min()} a {df_2026["Data"].max()}')
    print(f'✓ Coordenadas únicas: {df_2026[["Latitude", "Longitude"]].drop_duplicates().shape[0]:,}')

    return df_2026

def construir_grafo_otimizado(df_2026, usar_grafo=True):
    '''
    Constrói o grafo de forma otimizada para grandes volumes de dados.
    '''
    
    if not usar_grafo:
        print('\n[INFO] Grafo desabilitado - pulando construção')
        return None
    
    print(f'\n=== Construindo Grafo Otimizado ===')
    inicio = time.time()
    
    grafo = ClusterGraph()
    
    threshold_km = 100.0
    threshold_dias = 14

    print(f'Parâmetros: threshold_km={threshold_km}, threshold_dias={threshold_dias}')

    # ⚠️ REMOVER parâmetros se não existirem no método
    grafo.construir_grafo_dataframe(
        df_2026,
        threshold_km=threshold_km,
        threshold_dias=threshold_dias,
        usar_temporal=True,
        usar_espacial=True
    )
    
    tempo_construcao = time.time() - inicio
    print(f'✓ Grafo construído em {tempo_construcao:.2f}s')
    
    return grafo

def prever_dados(modelo, df_2026, usar_grafo=True, grafo=None):
    '''
    Aplica o modelo para prever RiscoFogo, opcionalmente usando features do grafo.
    '''

    print(f'\n=== Preparando Dados para Previsão ===')
    inicio_total = time.time()
    
    # ✅ USAR FUNÇÃO CENTRALIZADA COM PARÂMETROS CORRETOS
    df_preparado, _ = preparar_para_predicao(
        df_2026,
        usar_grafo=usar_grafo,
        grafo=grafo,
        label_encoders=modelo.get('label_encoders')
    )
    
    # Obter features na ordem correta
    feature_names = modelo.get('feature_names', [])
    
    # Adicionar features faltando com valor 0
    for feature in feature_names:
        if feature not in df_preparado.columns:
            df_preparado[feature] = 0
    
    X_pred = df_preparado[feature_names]
    
    # Fazer previsão
    try:
        if 'modelo' in modelo:
            risco_previsto = modelo['modelo'].predict(X_pred)
        else:
            raise ValueError('Modelo não contém "modelo"')
        
        print(f'✓ Previsão realizada!')
        print(f'  Risco - Min: {risco_previsto.min():.2f}, Max: {risco_previsto.max():.2f}, Média: {risco_previsto.mean():.2f}')
        
    except Exception as e:
        print(f'Erro na previsão: {e}')
        import traceback
        traceback.print_exc()
        risco_previsto = np.full(len(df_2026), 0.5)
    
    # Normalização
    risco_min, risco_max = risco_previsto.min(), risco_previsto.max()
    
    if risco_max <= 1.0:
        risco_normalizado = (risco_previsto * 100).clip(0, 100)
    elif risco_min < 0 or risco_max > 100:
        if risco_max > risco_min:
            risco_normalizado = ((risco_previsto - risco_min) / (risco_max - risco_min)) * 100
        else:
            risco_normalizado = np.full(len(risco_previsto), 50.0)
    else:
        risco_normalizado = risco_previsto.clip(0, 100)
    
    df_2026['RiscoFogo'] = np.round(risco_normalizado).astype(int)
    
    # Ajustar variáveis baseado no RiscoFogo
    df_2026['DiaSemChuva'] = (df_2026['DiaSemChuva'] * (1 + df_2026['RiscoFogo'] / 100 * 0.3)).astype(int)
    df_2026['Precipitacao'] = df_2026['Precipitacao'] * (1 - df_2026['RiscoFogo'] / 100 * 0.2)
    df_2026['FRP'] = df_2026['FRP'] * (1 + df_2026['RiscoFogo'] / 100 * 1.5)
    
    tempo_total = time.time() - inicio_total
    print(f'✓ Previsão concluída em {tempo_total:.2f}s')
    
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
    print(f'\n✓ Previsão salva em "{nome_arquivo}"')
    print(f'  Total de registros: {len(df_final):,}')
    
    # Estatísticas
    print(f'\nEstatísticas do RiscoFogo:')
    print(f'  Média: {df_final["RiscoFogo"].mean():.2f}')
    print(f'  Mediana: {df_final["RiscoFogo"].median():.2f}')
    print(f'  Min: {df_final["RiscoFogo"].min()}')
    print(f'  Max: {df_final["RiscoFogo"].max()}')

def main(usar_grafo=True, total_registros=None):
    '''
    Função principal para gerar previsões para 2026.
    '''
    print('='*80)
    print('GERAÇÃO DE PREVISÕES PARA 2026 - RISCO DE INCÊNDIO NO CERRADO')
    print(f'Modo: {"COM" if usar_grafo else "SEM"} Grafo')
    print('='*80)
    
    inicio_geral = time.time()
    
    # 1. Carregar modelo
    print('\n[1/6] Carregando modelo...')
    modelo = carregar_modelo()
    if modelo is None:
        print('Erro: Não foi possível carregar o modelo. Abortando.')
        return
    
    # 2. Carregar dados históricos
    print('\n[2/6] Carregando dados históricos...')
    df_historico = connection()
    
    # 3. Analisar padrões
    print('\n[3/6] Analisando padrões históricos...')
    padroes = analisar_padroes_historicos(df_historico)
    
    # 4. Gerar dados 2026
    print('\n[4/6] Gerando dados para 2026...')
    df_2026 = gerar_dados_2026(padroes, total_registros)
    
    # 5. Construir grafo
    grafo = None
    if usar_grafo:
        print('\n[5/6] Construindo grafo...')
        grafo = construir_grafo_otimizado(df_2026, usar_grafo=True)
    else:
        print('\n[5/6] Grafo desabilitado')
    
    # 6. Fazer previsões
    print('\n[6/6] Gerando previsões...')
    df_previsao = prever_dados(modelo, df_2026, usar_grafo=usar_grafo, grafo=grafo)
    
    # 7. Salvar resultados
    sufixo = '_com_grafo' if usar_grafo else '_sem_grafo'
    nome_arquivo = f'./source/previsao_2026{sufixo}.csv'
    salvar_previsao(df_previsao, nome_arquivo)
    
    tempo_total = time.time() - inicio_geral
    print(f'\n{"="*80}')
    print(f'PROCESSO CONCLUÍDO EM {tempo_total:.2f}s ({tempo_total/60:.2f} minutos)')
    print(f'{"="*80}')
    
    return df_previsao

if __name__ == '__main__':
    df_resultado = main(usar_grafo=True)