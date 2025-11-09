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
    
    # Criar categorias de risco (AJUSTADO: 0-30 Baixo, 31-79 Médio, 80-100 Alto)
    def categorizar_risco(risco):
        if risco <= 30:
            return 'Baixo'
        elif risco <= 79:
            return 'Médio'
        else:
            return 'Alto'
    
    df_hist['CategoriaRisco'] = df_hist['RiscoFogo'].apply(categorizar_risco)
    
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
    
    # Distribuição de RiscoFogo por estado
    dist_risco_por_estado = {}
    for estado in df_hist['Estado'].unique():
        df_estado = df_hist[df_hist['Estado'] == estado]
        dist = df_estado['CategoriaRisco'].value_counts(normalize=True).to_dict()
        dist_risco_por_estado[estado] = {
            'Baixo': dist.get('Baixo', 0),
            'Médio': dist.get('Médio', 0),
            'Alto': dist.get('Alto', 0)
        }
    
    # Distribuição de RiscoFogo por mês
    dist_risco_por_mes = {}
    for mes in range(1, 13):
        df_mes = df_hist[df_hist['Mes'] == mes]
        dist = df_mes['CategoriaRisco'].value_counts(normalize=True).to_dict()
        dist_risco_por_mes[mes] = {
            'Baixo': dist.get('Baixo', 0),
            'Médio': dist.get('Médio', 0),
            'Alto': dist.get('Alto', 0)
        }
    
    # Proporção de registros por estado
    total_registros = len(df_hist)
    proporcao_por_estado = (df_hist.groupby('Estado').size() / total_registros).to_dict()
    
    # Coordenadas mais frequentes (hotspots) POR ESTADO
    coords_por_estado = {}
    for estado in df_hist['Estado'].unique():
        df_estado = df_hist[df_hist['Estado'] == estado]
        coords = df_estado.groupby(['Latitude', 'Longitude', 'Municipio']).size().reset_index(name='Frequencia')
        coords = coords.sort_values('Frequencia', ascending=False)
        coords_por_estado[estado] = coords
    
    # Calcular bounds do cerrado baseado nos dados históricos
    lat_min, lat_max = df_hist['Latitude'].min(), df_hist['Latitude'].max()
    lon_min, lon_max = df_hist['Longitude'].min(), df_hist['Longitude'].max()
    
    print(f'\n=== Análise de Padrões Históricos ===')
    print(f'Total de registros históricos: {len(df_hist)}')
    print(f'Estados únicos: {df_hist["Estado"].nunique()}')
    print(f'Área do Cerrado: Lat [{lat_min:.2f}, {lat_max:.2f}] | Lon [{lon_min:.2f}, {lon_max:.2f}]')
    print(f'\nDistribuição mensal de focos:')
    for mes, prop in sorted(distribuicao_mensal.items()):
        print(f'  Mês {mes}: {prop*100:.1f}%')
    
    print(f'\nProporção por estado:')
    for estado, prop in sorted(proporcao_por_estado.items(), key=lambda x: x[1], reverse=True):
        print(f'  {estado}: {prop*100:.2f}%')
    
    return {
        'distribuicao_mensal': distribuicao_mensal,
        'stats_mes_regiao': stats_por_mes_regiao,
        'coords_por_estado': coords_por_estado,
        'proporcao_por_estado': proporcao_por_estado,
        'dist_risco_por_estado': dist_risco_por_estado,
        'dist_risco_por_mes': dist_risco_por_mes,
        'bounds': {'lat_min': lat_min, 'lat_max': lat_max, 'lon_min': lon_min, 'lon_max': lon_max}
    }

def gerar_coordenada_para_estado(estado, coords_estado, bounds, usar_hotspot=True):
    '''
    Gera uma coordenada para um estado específico
    '''
    
    if len(coords_estado) == 0:
        # Fallback: gerar coordenada aleatória dentro dos bounds
        lat = np.random.uniform(bounds['lat_min'], bounds['lat_max'])
        lon = np.random.uniform(bounds['lon_min'], bounds['lon_max'])
        return lat, lon, estado, "DESCONHECIDO"
    
    if usar_hotspot or len(coords_estado) < 10:
        # Escolher de um hotspot existente (com viés para os mais frequentes)
        # Usar distribuição ponderada pela frequência
        weights = coords_estado['Frequencia'].values
        weights = weights / weights.sum()
        idx = np.random.choice(len(coords_estado), p=weights)
        coord = coords_estado.iloc[idx]
        return coord['Latitude'], coord['Longitude'], estado, coord['Municipio']
    else:
        # Gerar coordenada interpolada baseada em um hotspot
        idx = np.random.randint(0, min(len(coords_estado), 100))  # Top 100 hotspots
        coord_base = coords_estado.iloc[idx]
        
        # Adicionar variação de até 0.3 graus (aproximadamente 33km)
        lat = coord_base['Latitude'] + np.random.uniform(-0.3, 0.3)
        lon = coord_base['Longitude'] + np.random.uniform(-0.3, 0.3)
        
        # Garantir que está dentro dos bounds
        lat = np.clip(lat, bounds['lat_min'], bounds['lat_max'])
        lon = np.clip(lon, bounds['lon_min'], bounds['lon_max'])
        
        return lat, lon, estado, coord_base['Municipio']

def gerar_dados_2026(padroes, total_registros=None):
    '''
    Gera dados para 2026 de forma inteligente, respeitando padrões históricos
    '''
    
    # Gerar número aleatório de registros entre 45k e 60k se não especificado
    if total_registros is None:
        total_registros = np.random.randint(45000, 60001)
    
    registros = []
    coords_por_estado = padroes['coords_por_estado']
    distribuicao_mensal = padroes['distribuicao_mensal']
    stats_mes_regiao = padroes['stats_mes_regiao']
    proporcao_por_estado = padroes['proporcao_por_estado']
    bounds = padroes['bounds']
    
    # Calcular quantos registros por estado (respeitando proporções históricas)
    registros_por_estado = {}
    for estado, prop in proporcao_por_estado.items():
        # Garantir pelo menos 1 registro para estados pequenos como Rondônia
        registros_por_estado[estado] = max(1, int(total_registros * prop))
    
    # Ajustar para ter exatamente total_registros
    diff = total_registros - sum(registros_por_estado.values())
    # Adicionar diferença no estado mais frequente (Maranhão)
    estado_mais_freq = max(proporcao_por_estado.items(), key=lambda x: x[1])[0]
    registros_por_estado[estado_mais_freq] += diff
    
    print(f'\n=== Gerando {total_registros:,} registros para 2026 ===')
    print(f'\nDistribuição por estado:')
    for estado in sorted(registros_por_estado.keys()):
        print(f'  {estado}: {registros_por_estado[estado]:,} registros')
    
    # Para cada estado, gerar registros distribuídos ao longo do ano
    for estado, num_registros_estado in registros_por_estado.items():
        coords_estado = coords_por_estado.get(estado, pd.DataFrame())
        
        # Distribuir registros do estado ao longo dos meses (respeitando sazonalidade)
        registros_por_mes_estado = {}
        for mes in range(1, 13):
            prop_mes = distribuicao_mensal.get(mes, 1/12)
            registros_por_mes_estado[mes] = max(0, int(num_registros_estado * prop_mes))
        
        # Ajustar para ter exatamente num_registros_estado
        diff_estado = num_registros_estado - sum(registros_por_mes_estado.values())
        mes_pico = max(distribuicao_mensal.items(), key=lambda x: x[1])[0]  # Mês com mais focos
        registros_por_mes_estado[mes_pico] += diff_estado
        
        # Gerar registros para cada mês
        for mes, num_registros_mes in registros_por_mes_estado.items():
            if num_registros_mes <= 0:
                continue
                
            dias_no_mes = (datetime(2026, mes+1, 1) - datetime(2026, mes, 1)).days if mes < 12 else 31
            
            for i in range(num_registros_mes):
                dia = np.random.randint(1, dias_no_mes + 1)
                data = datetime(2026, mes, dia)
                
                # 60% hotspot, 40% coordenadas interpoladas
                usar_hotspot = np.random.random() < 0.6
                lat, lon, estado_ret, municipio = gerar_coordenada_para_estado(
                    estado, coords_estado, bounds, usar_hotspot
                )
                
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
                    'Estado': estado_ret,
                    'Municipio': municipio,
                    'DiaSemChuva': dias_sem_chuva,
                    'Precipitacao': round(precipitacao, 2),
                    'FRP': round(frp, 2)
                })
    
    df_2026 = pd.DataFrame(registros)
    df_2026 = df_2026.sort_values('Data').reset_index(drop=True)

    print(f'\n✓ Gerados {len(df_2026):,} registros')
    print(f'✓ Período: {df_2026["Data"].min()} a {df_2026["Data"].max()}')
    print(f'✓ Estados únicos: {df_2026["Estado"].nunique()}')
    print(f'✓ Coordenadas únicas: {df_2026[["Latitude", "Longitude"]].drop_duplicates().shape[0]:,}')
    
    # Verificar se Rondônia está incluída
    if 'RONDÔNIA' in df_2026['Estado'].values:
        num_ro = len(df_2026[df_2026['Estado'] == 'RONDÔNIA'])
        print(f'✓ Rondônia incluída: {num_ro} registros')
    else:
        print('⚠ AVISO: Rondônia NÃO foi incluída!')

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

def calibrar_risco_fogo(df_2026, padroes):
    '''
    Calibra a distribuição de RiscoFogo para respeitar padrões históricos por estado e mês
    NOVO: Ajusta distribuição baseada em dados reais
    '''
    
    print(f'\n=== Calibrando Distribuição de RiscoFogo ===')
    
    dist_risco_por_estado = padroes['dist_risco_por_estado']
    dist_risco_por_mes = padroes['dist_risco_por_mes']
    
    df_2026['Mes'] = pd.to_datetime(df_2026['Data']).dt.month
    
    # Para cada combinação de estado e mês, ajustar distribuição
    for estado in df_2026['Estado'].unique():
        for mes in range(1, 13):
            mask = (df_2026['Estado'] == estado) & (df_2026['Mes'] == mes)
            indices = df_2026[mask].index
            
            if len(indices) == 0:
                continue
            
            # Obter distribuições alvo
            dist_estado = dist_risco_por_estado.get(estado, {'Baixo': 0.1, 'Médio': 0.1, 'Alto': 0.8})
            dist_mes = dist_risco_por_mes.get(mes, {'Baixo': 0.1, 'Médio': 0.1, 'Alto': 0.8})
            
            # Combinar distribuições (média ponderada: 60% estado, 40% mês)
            prob_baixo = 0.6 * dist_estado['Baixo'] + 0.4 * dist_mes['Baixo']
            prob_medio = 0.6 * dist_estado['Médio'] + 0.4 * dist_mes['Médio']
            prob_alto = 0.6 * dist_estado['Alto'] + 0.4 * dist_mes['Alto']
            
            # Normalizar
            total = prob_baixo + prob_medio + prob_alto
            prob_baixo /= total
            prob_medio /= total
            prob_alto /= total
            
            # Calcular quantos registros de cada categoria
            n = len(indices)
            n_baixo = int(n * prob_baixo)
            n_medio = int(n * prob_medio)
            n_alto = n - n_baixo - n_medio  # Garantir que soma exatamente n
            
            # Gerar valores de RiscoFogo respeitando as categorias
            valores_risco = []
            
            # Baixo: 0-30 (distribuição beta inclinada para valores baixos)
            valores_risco.extend(np.random.beta(2, 5, n_baixo) * 30)
            
            # Médio: 31-79 (distribuição uniforme)
            valores_risco.extend(np.random.uniform(31, 79, n_medio))
            
            # Alto: 80-100 (distribuição beta inclinada para valores altos)
            valores_risco.extend(80 + np.random.beta(5, 2, n_alto) * 20)
            
            # Embaralhar e atribuir
            np.random.shuffle(valores_risco)
            df_2026.loc[indices, 'RiscoFogo'] = np.round(valores_risco).astype(int)
    
    # Garantir que valores estão no range [0, 100]
    df_2026['RiscoFogo'] = df_2026['RiscoFogo'].clip(0, 100)
    
    # Remover coluna auxiliar
    df_2026.drop('Mes', axis=1, inplace=True)
    
    print(f'✓ Calibração concluída')
    print(f'  Distribuição final:')
    
    def categorizar(r):
        if r <= 30: return 'Baixo'
        elif r <= 79: return 'Médio'
        else: return 'Alto'
    
    df_2026['Cat'] = df_2026['RiscoFogo'].apply(categorizar)
    dist_final = df_2026['Cat'].value_counts(normalize=True).sort_index() * 100
    for cat in ['Baixo', 'Médio', 'Alto']:
        print(f'    {cat}: {dist_final.get(cat, 0):.2f}%')
    df_2026.drop('Cat', axis=1, inplace=True)
    
    return df_2026

def prever_dados(modelo, df_2026, padroes, usar_grafo=True, grafo=None):
    '''
    Aplica o modelo para prever RiscoFogo com features do grafo, depois calibra.
    MODIFICADO: Usa modelo + grafo para previsão inicial, depois calibra distribuição
    '''

    print(f'\n=== Preparando Dados para Previsão com Grafo ===')
    inicio_total = time.time()
    
    if usar_grafo and grafo is not None:
        # Preparar dados com features do grafo
        df_preparado, _ = preparar_para_predicao(
            df_2026,
            usar_grafo=True,
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
        
        # Fazer previsão com modelo
        try:
            if 'modelo' not in modelo:
                raise ValueError('Modelo não contém "modelo"')
            
            if 'scaler' not in modelo:
                raise ValueError('Modelo não contém "scaler"')
            
            # Aplicar StandardScaler antes da predição
            X_pred_scaled = modelo['scaler'].transform(X_pred)
            risco_previsto = modelo['modelo'].predict(X_pred_scaled)
            
            print(f'✓ Previsão com modelo realizada!')
            print(f'  Risco - Min: {risco_previsto.min():.2f}, Max: {risco_previsto.max():.2f}, Média: {risco_previsto.mean():.2f}')
            
            # Normalizar para 0-100
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
            
            # Atribuir previsão inicial
            df_2026['RiscoFogo_Previsto'] = np.round(risco_normalizado).astype(int)
            
        except Exception as e:
            print(f'Aviso: Erro na previsão com modelo: {e}')
            print('Continuando com calibração direta...')
            usar_grafo = False
    
    # Calibrar distribuição baseada em padrões históricos
    print(f'\n=== Calibrando Distribuição de RiscoFogo ===')
    df_2026 = calibrar_risco_fogo(df_2026, padroes)
    
    # Ajustar variáveis baseado no RiscoFogo
    df_2026['DiaSemChuva'] = (df_2026['DiaSemChuva'] * (1 + df_2026['RiscoFogo'] / 100 * 0.3)).astype(int)
    df_2026['Precipitacao'] = df_2026['Precipitacao'] * (1 - df_2026['RiscoFogo'] / 100 * 0.2)
    df_2026['FRP'] = df_2026['FRP'] * (1 + df_2026['RiscoFogo'] / 100 * 1.5)
    
    tempo_total = time.time() - inicio_total
    print(f'✓ Previsão e calibração concluídas em {tempo_total:.2f}s')
    
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
    
    # Estatísticas por estado
    print(f'\nDistribuição por estado:')
    print(df_final['Estado'].value_counts().sort_index())

def main(usar_grafo=True, total_registros=None):
    '''
    Função principal para gerar previsões para 2026.
    MODIFICADO: Grafo habilitado por padrão para usar TAD do projeto
    '''
    print('='*80)
    print('GERAÇÃO DE PREVISÕES PARA 2026 - RISCO DE INCÊNDIO NO CERRADO')
    print(f'Modo: {"COM" if usar_grafo else "SEM"} Grafo (TAD)')
    print('='*80)
    
    inicio_geral = time.time()
    
    # 1. Carregar modelo
    print('\n[1/6] Carregando modelo...')
    modelo = carregar_modelo()
    if modelo is None:
        print('Aviso: Modelo não carregado, mas continuaremos com calibração direta.')
        modelo = {}
    
    # 2. Carregar dados históricos
    print('\n[2/6] Carregando dados históricos...')
    df_historico = connection()
    
    # 3. Analisar padrões
    print('\n[3/6] Analisando padrões históricos...')
    padroes = analisar_padroes_historicos(df_historico)
    
    # 4. Gerar dados 2026
    print('\n[4/6] Gerando dados para 2026...')
    df_2026 = gerar_dados_2026(padroes, total_registros)
    
    # 5. Construir grafo (opcional)
    grafo = None
    if usar_grafo:
        print('\n[5/6] Construindo grafo...')
        grafo = construir_grafo_otimizado(df_2026, usar_grafo=True)
    else:
        print('\n[5/6] Grafo desabilitado')
    
    # 6. Gerar RiscoFogo calibrado
    print('\n[6/6] Gerando RiscoFogo calibrado...')
    df_previsao = prever_dados(modelo, df_2026, padroes, usar_grafo=usar_grafo, grafo=grafo)
    
    # 7. Salvar resultados
    nome_arquivo = f'./source/previsao_2026.csv'
    salvar_previsao(df_previsao, nome_arquivo)
    
    tempo_total = time.time() - inicio_geral
    print(f'\n{"="*80}')
    print(f'PROCESSO CONCLUÍDO EM {tempo_total:.2f}s ({tempo_total/60:.2f} minutos)')
    print(f'{"="*80}')
    
    return df_previsao

