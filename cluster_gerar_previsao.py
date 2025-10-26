import pandas as pd
import pickle
import numpy as np
from sklearn.cluster import KMeans
from data.connection import connection
from cluster_utils import preparar_dados

def carregar_modelo(filepath='modelo_cluster.pkl'):
    """
    Carrega o modelo salvo em um arquivo .pkl
    """
    with open(filepath, 'rb') as f:
        modelo_cluster = pickle.load(f)
    return modelo_cluster

def amostrar_regioes_por_cluster(df, n_clusters):
    """
    Amostra regiões representativas usando K-Means para dividir o Cerrado em clusters geográficos.
    Retorna os centroides como regiões representativas.
    """
    
    seed = 4224
    
    regioes = df[['Latitude', 'Longitude']].drop_duplicates()
    if len(regioes) < n_clusters:
        return regioes

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    regioes['Cluster'] = kmeans.fit_predict(regioes[['Latitude', 'Longitude']])
    
    regioes_representativas = []
    for cluster in range(n_clusters):
        cluster_data = regioes[regioes['Cluster'] == cluster]
        n_samples = max(1, len(cluster_data) // 10)
        samples = cluster_data.sample(n=min(n_samples, len(cluster_data)), random_state=seed)
        regioes_representativas.append(samples[['Latitude', 'Longitude']])
    
    return pd.concat(regioes_representativas, ignore_index=True)

def prever_e_salvar(df, modelo_cluster, dias_previsao, regioes_por_dia=50, agregar_por_dia=False):
    """
    Realiza a previsão usando dados reais do histórico baseado em mês/ano
    """
    
    seed = 4224
    
    if df['Data'].dtype == 'object':
        df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    
    if df['Data'].isna().any():
        raise ValueError("A coluna 'Data' contém valores inválidos.")
    
    # PREVISÃO PARA 2026
    data_inicio = pd.Timestamp('2026-01-01')
    data_fim = data_inicio + pd.Timedelta(days=dias_previsao - 1)
    
    novas_datas = pd.date_range(start=data_inicio, end=data_fim, freq='D')
    
    # Selecionar regiões representativas
    regioes = amostrar_regioes_por_cluster(df, n_clusters=12)
    
    if len(regioes) > regioes_por_dia:
        regioes = regioes.sample(n=regioes_por_dia, random_state=seed)
    
    # Criar DataFrame de previsão
    df_previsao = pd.DataFrame({'Data': novas_datas})
    df_previsao = df_previsao.merge(regioes, how='cross')
    
    # Extrair mês e ano para buscar dados históricos correspondentes
    df_previsao['Mes'] = df_previsao['Data'].dt.month
    df_previsao['Ano'] = 2026
    
    df['Mes'] = df['Data'].dt.month
    df['Ano'] = df['Data'].dt.year
    
    # Para cada linha da previsão, buscar dados reais do mesmo mês/região de anos anteriores
    resultado_final = []
    
    for _, row_prev in df_previsao.iterrows():
        # Buscar dados históricos do mesmo mês e região próxima
        dados_historicos = df[
            (df['Mes'] == row_prev['Mes']) &
            (np.abs(df['Latitude'] - row_prev['Latitude']) < 0.5) &
            (np.abs(df['Longitude'] - row_prev['Longitude']) < 0.5)
        ]
        
        if len(dados_historicos) > 0:
            # Pegar uma amostra aleatória dos dados históricos
            amostra = dados_historicos.sample(n=1, random_state=seed)
            
            nova_linha = {
                'Data': row_prev['Data'],
                'Latitude': row_prev['Latitude'],
                'Longitude': row_prev['Longitude'],
                'DiaSemChuva': int(amostra['DiaSemChuva'].values[0]),
                'Precipitacao': round(float(amostra['Precipitacao'].values[0]), 2),
                'FRP': round(float(amostra['FRP'].values[0]), 2)
            }
        else:
            # Fallback: usar médias do mês se não encontrar dados próximos
            dados_mes = df[df['Mes'] == row_prev['Mes']]
            nova_linha = {
                'Data': row_prev['Data'],
                'Latitude': row_prev['Latitude'],
                'Longitude': row_prev['Longitude'],
                'DiaSemChuva': int(dados_mes['DiaSemChuva'].mean()),
                'Precipitacao': round(float(dados_mes['Precipitacao'].mean()), 2),
                'FRP': round(float(dados_mes['FRP'].mean()), 2)
            }
        
        resultado_final.append(nova_linha)
    
    df_previsao = pd.DataFrame(resultado_final)
    
    # Preparar dados e fazer previsão
    df_preparado, _ = preparar_dados(df_previsao, modelo_cluster)
    
    feature_names = modelo_cluster['feature_names']
    for col in feature_names:
        if col not in df_preparado.columns:
            df_preparado[col] = 0
    
    df_preparado = df_preparado[feature_names]
    
    # PREVISÃO
    X_scaled = modelo_cluster['scaler'].transform(df_preparado)
    previsoes = modelo_cluster['modelo'].predict(X_scaled)
    
    # Aplicar o RiscoFogo diretamente
    df_previsao['RiscoFogo'] = np.abs(previsoes).round(0).astype(int)
    
    if agregar_por_dia:
        df_final = df_previsao.groupby('Data').agg({
            'RiscoFogo': 'mean',
            'Latitude': 'mean',
            'Longitude': 'mean',
            'DiaSemChuva': 'mean',
            'Precipitacao': 'mean',
            'FRP': 'mean'
        }).reset_index()
        df_final['RiscoFogo'] = df_final['RiscoFogo'].round(0).astype(int)
        df_final['DiaSemChuva'] = df_final['DiaSemChuva'].round(0).astype(int)
    else:
        df_final = df_previsao[['Data', 'Latitude', 'Longitude', 'RiscoFogo', 'DiaSemChuva', 'Precipitacao', 'FRP']]
    
    output_filepath = f'previsao_2026_{dias_previsao}_dias.csv'
    
    with open(output_filepath, 'w', newline='', encoding='utf-8') as f:
        df_final.to_csv(f, index=False)
        print(f"Arquivo salvo em: {output_filepath}")
        print(f"Total de linhas: {len(df_final)}")
        print(f"Período: {df_final['Data'].min()} até {df_final['Data'].max()}")

# Carregar os dados e o modelo
df = connection()
modelo_cluster = carregar_modelo()

# Prever para 2026
prever_e_salvar(df, modelo_cluster, dias_previsao=365, regioes_por_dia=20, agregar_por_dia=False)