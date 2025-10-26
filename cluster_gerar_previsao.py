import pandas as pd
import numpy as np
import pickle
from datetime import timedelta

from data.connection import connection
from cluster_predicao import preparar_features

def gerar_previsoes_futuras(dias_futuro):
    '''
    Gera previsões para os próximos dias
    '''
    
    print('Carregando modelo...')
    with open('modelo_cluster.pkl', 'rb') as f:
        modelo_cluster = pickle.load(f)
    
    modelo = modelo_cluster['modelo']
    scaler = modelo_cluster['scaler']
    feature_names = modelo_cluster['feature_names']
    print('✓ Modelo carregado')
    
    print('\nCarregando dados base...')
    df_base = connection()
    df_base['Data'] = pd.to_datetime(df_base['Data'])
    print('✓ Dados carregados')
    
    ultima_data = df_base['Data'].max()
    datas_futuras = pd.date_range(
        start=ultima_data + timedelta(days=1),
        periods=dias_futuro,
        freq='D'
    )
    
    localizacoes = df_base[['Estado', 'Municipio', 'Latitude', 'Longitude']].drop_duplicates()
    total_registros = len(localizacoes) * dias_futuro
    
    print(f'\nGerando previsões:')
    print(f'   • {len(localizacoes)} locais')
    print(f'   • {dias_futuro} dias')
    print(f'   • {total_registros:,} registros totais\n')
    
    # Calcular medianas por localização antecipadamente
    print('Calculando medianas por localização...')
    medianas_loc = df_base.groupby(['Estado', 'Municipio']).agg({
        'DiaSemChuva': 'median',
        'Precipitacao': 'median',
        'FRP': 'median',
        'Latitude': 'first',
        'Longitude': 'first'
    }).reset_index()
    
    # Outras colunas numéricas
    outras_cols = [col for col in df_base.columns 
                   if col not in ['DataHora', 'RiscoFogo', 'Data', 'Estado', 'Municipio', 
                                  'DiaSemChuva', 'Precipitacao', 'FRP', 'Latitude', 'Longitude']
                   and df_base[col].dtype in ['float64', 'int64']]
    
    if outras_cols:
        outras_medianas = df_base.groupby(['Estado', 'Municipio'])[outras_cols].median().reset_index()
        medianas_loc = medianas_loc.merge(outras_medianas, on=['Estado', 'Municipio'])
    
    print('✓ Medianas calculadas\n')
    
    # Criar todas as combinações de uma vez
    print('Criando combinações de dados...')
    previsoes_list = []
    contador = 0
    
    for idx, loc in medianas_loc.iterrows():
        for data in datas_futuras:
            linha = {
                'Data': data,
                'Estado': loc['Estado'],
                'Municipio': loc['Municipio'],
                'Latitude': loc['Latitude'],
                'Longitude': loc['Longitude'],
                'DiaSemChuva': loc['DiaSemChuva'],
                'Precipitacao': loc['Precipitacao'],
                'FRP': loc['FRP']
            }
            
            for col in outras_cols:
                linha[col] = loc[col]
            
            previsoes_list.append(linha)
            contador += 1
            
            if contador % 100 == 0:
                porcentagem = (contador / total_registros) * 100
                print(f'   Processando: {contador:,}/{total_registros:,} linhas ({porcentagem:.1f}%)')
    
    df_previsoes = pd.DataFrame(previsoes_list)
    print(f'{len(df_previsoes):,} combinações criadas\n')
    
    datas_originais = df_previsoes[['Data', 'Estado', 'Municipio', 'Latitude', 'Longitude']].copy()
    
    print('Preparando features...')
    df_pred_preparado, _ = preparar_features(df_previsoes)
    
    for col in feature_names:
        if col not in df_pred_preparado.columns:
            df_pred_preparado[col] = 0
    
    df_pred_preparado = df_pred_preparado[feature_names]
    print('Features preparadas\n')
    
    print('Fazendo previsões com o modelo...')
    X_pred_scaled = scaler.transform(df_pred_preparado)
    previsoes_risco = modelo.predict(X_pred_scaled)
    print('Previsões concluídas\n')

    print('Montando resultado final...')
    df_resultado = datas_originais.copy()
    df_resultado['RiscoFogo'] = previsoes_risco
    df_resultado['DiaSemChuva'] = df_previsoes['DiaSemChuva'].values
    df_resultado['Precipitacao'] = df_previsoes['Precipitacao'].values
    df_resultado['FRP'] = df_previsoes['FRP'].values
    
    df_resultado['Data'] = df_resultado['Data'].dt.strftime('%Y-%m-%d')
    
    print('Salvando CSV...')
    df_resultado.to_csv('previsoes_queimadas.csv', index=False)
    print(f'\nConcluído! {len(df_resultado):,} previsões salvas em previsoes_queimadas.csv\n')
    
    return df_resultado


gerar_previsoes_futuras(90)