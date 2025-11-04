import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime
from cluster.cluster_utils import preparar_dados
from cluster.cluster import criacao_variaveis_mes
from data.connection import connection
from models.ClusterGraph import ClusterGraph

def carregar_modelo(caminho_modelo='./modelo_completo_grafo.pkl'):
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
    
    print(f'\n=== Gerando {total_registros:,} registros para 2026 ===')
    
    # Para cada mês, gerar registros distribuídos ao longo dos dias
    for mes in range(1, 13):
        num_registros_mes = registros_por_mes[mes]
        dias_no_mes = (datetime(2026, mes+1, 1) - datetime(2026, mes, 1)).days if mes < 12 else 31
        
        print(f'Mês {mes}: {num_registros_mes} registros')
        
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

    print(f'\n✓ Gerados {len(df_2026):,} registros')
    print(f'✓ Período: {df_2026["Data"].min()} a {df_2026["Data"].max()}')
    print(f'✓ Coordenadas únicas: {df_2026[["Latitude", "Longitude"]].drop_duplicates().shape[0]:,}')

    return df_2026

def construir_grafo_otimizado(df_2026, usar_grafo=True):
    '''
    Constrói o grafo de forma otimizada para grandes volumes de dados.
    
    Otimizações aplicadas com grafo:
    - Limitar conexões por vértice (max_conexoes_por_vertice)
    - Usar grid espacial para reduzir comparações
    - Desabilitar progresso detalhado para grandes datasets
    '''
    
    if not usar_grafo:
        print('\n[INFO] Grafo desabilitado - pulando construção')
        return None
    
    print(f'\n=== Construindo Grafo Otimizado ===')
    inicio = time.time()
    
    # Criar grafo com parâmetros otimizados
    grafo = ClusterGraph()
    
    # Parâmetros otimizados para performance
    threshold_km = 100.0  # Aumentar threshold para reduzir número de arestas
    threshold_dias = 14   # Aumentar janela temporal
    max_conexoes = 30     # Limitar conexões por vértice

    print(f'Parâmetros: threshold_km={threshold_km}, threshold_dias={threshold_dias}, max_conexoes={max_conexoes}')

    # Construir grafo (usa método otimizado internamente)
    grafo.construir_grafo_dataframe(
        df_2026,
        threshold_km=threshold_km,
        threshold_dias=threshold_dias,
        usar_temporal=True,
        usar_espacial=True,
        max_conexoes_por_vertice=max_conexoes,
        mostrar_progresso=True
    )
    
    return grafo

def extrair_features_grafo_otimizado(grafo, df_2026):
    '''
    Extrai features do grafo de forma otimizada.
    '''
    
    if grafo is None:
        print('\n[INFO] Grafo não disponível - pulando extração de features')
        return df_2026
    
    print(f'\n=== Extraindo Features do Grafo ===')
    inicio = time.time()
    
    # Extrair features (usa cache interno)
    df_com_features = grafo.extrair_features_dataframe(df_2026, mostrar_progresso=True)
    
    tempo_extracao = time.time() - inicio
    print(f'✓ Features extraídas em {tempo_extracao:.2f}s')
    
    return df_com_features

def preparar_dados_com_grafo(df, modelo_cluster):
    '''
    Prepara os dados para o modelo, tratando valores desconhecidos nos LabelEncoders.
    Esta é uma versão modificada que lida com valores não vistos durante o treinamento.
    '''
    
    df_copy = df.copy()
    
    # Guardar Estado e Municipio originais antes de codificar
    estado_original = df_copy['Estado'].copy()
    municipio_original = df_copy['Municipio'].copy()
    
    # Criar variáveis dummy para os meses
    df_copy = criacao_variaveis_mes(df_copy)
    
    # Garantir que a coluna 'Data' esteja no formato datetime
    if df_copy['Data'].dtype == 'object':
        df_copy['Data'] = pd.to_datetime(df_copy['Data'], errors='coerce')
    
    # Adicionar colunas de ano e dia do ano
    df_copy['Ano'] = df_copy['Data'].dt.year
    df_copy['DiaAno'] = df_copy['Data'].dt.dayofyear
    
    # Codificar colunas categóricas, tratando valores desconhecidos
    if modelo_cluster and 'label_encoders' in modelo_cluster:
        label_encoders = modelo_cluster['label_encoders']
        
        # Codificar Estado
        if 'Estado' in df_copy.columns and 'Estado' in label_encoders:
            le_estado = label_encoders['Estado']
            # Mapear valores conhecidos, usar -1 para desconhecidos
            estado_encoded = []
            for estado in df_copy['Estado'].astype(str):
                if estado in le_estado.classes_:
                    estado_encoded.append(le_estado.transform([estado])[0])
                else:
                    # Usar o primeiro estado conhecido como fallback
                    estado_encoded.append(le_estado.transform([le_estado.classes_[0]])[0])
            df_copy['Estado_encoded'] = estado_encoded
        
        # Codificar Município
        if 'Municipio' in df_copy.columns and 'Municipio' in label_encoders:
            le_municipio = label_encoders['Municipio']
            # Mapear valores conhecidos, usar -1 para desconhecidos
            municipio_encoded = []
            for municipio in df_copy['Municipio'].astype(str):
                if municipio in le_municipio.classes_:
                    municipio_encoded.append(le_municipio.transform([municipio])[0])
                else:
                    # Usar o primeiro município conhecido como fallback
                    municipio_encoded.append(le_municipio.transform([le_municipio.classes_[0]])[0])
            df_copy['Municipio_encoded'] = municipio_encoded
    
    # Restaurar Estado e Municipio originais
    df_copy['Estado'] = estado_original
    df_copy['Municipio'] = municipio_original
    
    # Remover colunas desnecessárias para previsão (mas manter Estado e Municipio originais)
    colunas_remover = ['DataHora', 'Data']
    df_preparado = df_copy.drop(columns=[col for col in colunas_remover if col in df_copy.columns])
    
    return df_preparado

def prever_dados(modelo, df_2026, usar_grafo=True, grafo=None):
    '''
    Aplica o modelo para prever RiscoFogo, opcionalmente usando features do grafo.
    '''

    print(f'\n=== Preparando Dados para Previsão ===')
    inicio_total = time.time()
    
    # Se usar grafo, extrair features
    if usar_grafo and grafo is not None:
        df_2026 = extrair_features_grafo_otimizado(grafo, df_2026)
    
    # Preparar dados usando a função modificada
    df_preparado = preparar_dados_com_grafo(df_2026, modelo_cluster=modelo)
    
    # Obter features na ordem correta
    feature_names = modelo.get('feature_names', [])
    
    # Verificar quais features estão disponíveis
    features_disponiveis = [f for f in feature_names if f in df_preparado.columns]
    print(f'Features disponíveis: {len(features_disponiveis)}/{len(feature_names)}')
    
    if len(features_disponiveis) < len(feature_names):
        features_faltando = set(feature_names) - set(features_disponiveis)
        print(f'Aviso: Features faltando: {features_faltando}')
        # Adicionar features faltando com valor 0
        for feature in features_faltando:
            df_preparado[feature] = 0
        features_disponiveis = feature_names

    X_pred = df_preparado[features_disponiveis]
    
    # Fazer previsão do RiscoFogo
    try:
        if 'modelo' in modelo:
            # Modelo de regressão - prevê RiscoFogo diretamente
            risco_previsto = modelo['modelo'].predict(X_pred)
            print(f'✓ Previsão realizada com sucesso!')
            print(f'  Risco previsto - Min: {risco_previsto.min():.2f}, Max: {risco_previsto.max():.2f}, Média: {risco_previsto.mean():.2f}')
        elif 'kmeans' in modelo:
            # Modelo de clustering - mapeia clusters para risco
            clusters = modelo['kmeans'].predict(X_pred)
            if 'cluster_stats' in modelo:
                cluster_risk_map = {}
                for cluster_id, stats in modelo['cluster_stats'].items():
                    cluster_risk_map[cluster_id] = stats.get('RiscoFogo_mean', 0.5)
                risco_previsto = np.array([cluster_risk_map.get(c, 0.5) for c in clusters])
            else:
                max_cluster = clusters.max()
                risco_previsto = (clusters / max_cluster if max_cluster > 0 else 0.5)
            print(f'✓ Previsão realizada com sucesso!')
        else:
            raise ValueError('Modelo não contém "modelo" ou "kmeans"')
    except Exception as e:
        print(f'Erro na previsão: {e}')
        import traceback
        traceback.print_exc()
        risco_previsto = np.full(len(df_2026), 0.5)
    
    # Adicionar RiscoFogo ao dataframe
    # Normalizar valores previstos para faixa 0-100
    risco_min = risco_previsto.min()
    risco_max = risco_previsto.max()
    
    if risco_min < 0 or risco_max > 100:
        # Normalizar para 0-100 se os valores estão fora da faixa esperada
        if risco_max > risco_min:
            risco_normalizado = ((risco_previsto - risco_min) / (risco_max - risco_min)) * 100
        else:
            risco_normalizado = np.full(len(risco_previsto), 50.0)
        print(f'  Risco normalizado para 0-100 - Min: {risco_normalizado.min():.2f}, Max: {risco_normalizado.max():.2f}, Média: {risco_normalizado.mean():.2f}')
        df_2026['RiscoFogo'] = risco_normalizado
    elif risco_max <= 1.0:
        # Se está em escala 0-1, converter para 0-100
        df_2026['RiscoFogo'] = (risco_previsto * 100).clip(0, 100)
    else:
        # Já está em escala 0-100
        df_2026['RiscoFogo'] = risco_previsto.clip(0, 100)
    
    # Converter para inteiro
    df_2026['RiscoFogo'] = np.round(df_2026['RiscoFogo']).astype(int)
    
    # Ajustar sutilmente variáveis baseado no RiscoFogo previsto
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
    print(f'  Colunas: {list(df_final.columns)}')
    
    # Estatísticas resumidas
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
    if usar_grafo:
        print('Modo: COM Grafo (features espaciais e temporais)')
    else:
        print('Modo: SEM Grafo (apenas features originais)')
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
    
    # 5. Construir grafo (se habilitado)
    grafo = None
    if usar_grafo:
        print('\n[5/6] Construindo grafo...')
        grafo = construir_grafo_otimizado(df_2026, usar_grafo=True)
    else:
        print('\n[5/6] Grafo desabilitado - pulando construção')
    
    # 6. Fazer previsões
    print('\n[6/6] Gerando previsões...')
    df_previsao = prever_dados(modelo, df_2026, usar_grafo=usar_grafo, grafo=grafo)
    
    # 7. Salvar resultados
    sufixo = '_com_grafo' if usar_grafo else '_sem_grafo'
    nome_arquivo = f'previsao_2026{sufixo}.csv'
    salvar_previsao(df_previsao, nome_arquivo)
    
    tempo_total = time.time() - inicio_geral
    print(f'\n{"="*80}')
    print(f'PROCESSO CONCLUÍDO EM {tempo_total:.2f}s ({tempo_total/60:.2f} minutos)')
    print(f'{"="*80}')
    
    return df_previsao

if __name__ == '__main__':
    # Executar com grafo (padrão)
    # Para desabilitar o grafo e ter execução mais rápida, use: main(usar_grafo=False)
    df_resultado = main(usar_grafo=True)