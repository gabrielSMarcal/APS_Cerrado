import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from cluster.preparacao_dados import preparar_para_predicao, validar_features
from prev.cluster_predicao import treinar_modelo
from data.connection import connection 

import folium

# Configurações
OUT_DIR = './assets/avaliacao_outputs'
MODEL_PATH = './models/modelo_completo_grafo.pkl'
MARGEM = 10
SAMPLE_MAP = None

os.makedirs(OUT_DIR, exist_ok=True)

def carregar_modelo_ou_treinar(model_path=MODEL_PATH):
    '''
    Tenta carregar modelo de model_path. Se não existir, treina e salva.
    '''
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            modelo_cluster = pickle.load(f)
        print(f'Modelo carregado de {model_path}')
        return modelo_cluster

    print(f'Arquivo {model_path} não encontrado. Treinando modelo automaticamente...')
    df_hist = connection()
    if df_hist is None or len(df_hist) == 0:
        raise RuntimeError('Dados históricos vazios — não é possível treinar o modelo automaticamente.')

    modelo_cluster = treinar_modelo(df_hist, usar_grafo=True, salvar_modelo=True, caminho_modelo=model_path)

    return modelo_cluster


def preparar_e_prever_tudo(df_raw, modelo_cluster):
    '''
    Prepara dados e faz predições usando a função centralizada.
    '''
    df = df_raw.copy()
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df['Ano'] = df['Data'].dt.year

    # Usar a função centralizada de preparação
    usar_grafo = modelo_cluster.get('usar_grafo', False)
    grafo = modelo_cluster.get('grafo', None)
    label_encoders = modelo_cluster.get('label_encoders', None)
    
    df_preparado, _ = preparar_para_predicao(
        df,
        usar_grafo=usar_grafo,
        grafo=grafo,
        label_encoders=label_encoders
    )

    # Copiar colunas importantes para o resultado final
    keys_to_copy = ['Data', 'Latitude', 'Longitude', 'Estado', 'Municipio', 'RiscoFogo', 'Ano']
    df_preparado = df_preparado.reset_index(drop=True)
    df = df.reset_index(drop=True)
    
    for k in keys_to_copy:
        if k in df.columns and k not in df_preparado.columns:
            df_preparado[k] = df[k]

    # Obter features do modelo
    feature_names = modelo_cluster.get('feature_names', [])
    
    if not feature_names:
        raise RuntimeError('modelo_cluster não contém "feature_names".')
    
    # Validar e reordenar features
    if 'RiscoFogo' in df_preparado.columns:
        X_all = df_preparado.drop(columns=['RiscoFogo'])
    else:
        X_all = df_preparado.copy()
    
    X_all = validar_features(X_all, feature_names)

    # Normalizar
    scaler = modelo_cluster.get('scaler', None)
    if scaler is not None:
        X_all_scaled = scaler.transform(X_all)
    else:
        X_all_scaled = X_all.values

    # Predizer
    model = modelo_cluster.get('modelo')
    if model is None:
        raise RuntimeError('modelo_cluster não contém chave "modelo".')

    y_pred = model.predict(X_all_scaled)
    y_pred = np.clip(y_pred, 0, 100).round().astype(int)
    df_preparado['Risco_pred'] = y_pred

    return df_preparado, feature_names


def calcular_metricas_por_ano(df_eval, y_col='RiscoFogo'):
    anos = sorted(df_eval['Ano'].unique())
    metrics = []
    margin_metrics = []

    for a in anos:
        sub = df_eval[df_eval['Ano'] == a]
        if sub.empty:
            continue
        y_true = sub[y_col].astype(float)
        y_pred = sub['Risco_pred'].astype(float)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        metrics.append({'Ano': int(a), 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'n': len(sub)})
        acc_margin = (np.abs(y_true - y_pred) <= MARGEM).mean()
        margin_metrics.append({'Ano': int(a), 'accuracy_margin': acc_margin, 'n': len(sub)})

    metrics_df = pd.DataFrame(metrics).sort_values('Ano').reset_index(drop=True)
    margin_df = pd.DataFrame(margin_metrics).sort_values('Ano').reset_index(drop=True)
    return metrics_df, margin_df


def plot_e_salvar_metricas(metrics_df, margin_df, out_dir=OUT_DIR):
    # MAE
    plt.figure(figsize=(10,5))
    plt.bar(metrics_df['Ano'].astype(str), metrics_df['MAE'], color='skyblue')
    plt.title('MAE por Ano')
    plt.xlabel('Ano')
    plt.ylabel('MAE')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mae_por_ano.png'), dpi=150)
    plt.close()

    # RMSE
    plt.figure(figsize=(10,5))
    plt.bar(metrics_df['Ano'].astype(str), metrics_df['RMSE'], color='orange')
    plt.title('RMSE por Ano')
    plt.xlabel('Ano')
    plt.ylabel('RMSE')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rmse_por_ano.png'), dpi=150)
    plt.close()

    # R2
    plt.figure(figsize=(10,5))
    plt.bar(metrics_df['Ano'].astype(str), metrics_df['R2'], color='green')
    plt.title('R² por Ano')
    plt.xlabel('Ano')
    plt.ylabel('R²')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'r2_por_ano.png'), dpi=150)
    plt.close()

    # Acurácia por margem
    plt.figure(figsize=(10,5))
    plt.plot(margin_df['Ano'], margin_df['accuracy_margin'], marker='o')
    plt.title(f'Acurácia por margem ±{MARGEM} por Ano')
    plt.xlabel('Ano')
    plt.ylabel('Acurácia (0-1)')
    plt.ylim(0,1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'acuracia_margem_por_ano.png'), dpi=150)
    plt.close()

    # salvar CSVs
    metrics_df.to_csv(os.path.join(out_dir, 'metrics_continuas_por_ano.csv'), index=False)
    margin_df.to_csv(os.path.join(out_dir, 'metrics_margem_por_ano.csv'), index=False)

def carregar_dados_csvs(pasta_csvs):
    import re
    arquivos = [os.path.join(pasta_csvs, f) for f in os.listdir(pasta_csvs) if f.endswith('.csv')]
    dfs = []
    for arq in arquivos:
        df = pd.read_csv(arq)
        # garante que a coluna Data é datetime
        if 'DataHora' in df.columns:
            df['Data'] = pd.to_datetime(df['DataHora'], errors='coerce')
        elif 'Data' in df.columns:
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
        # extrai o ano
        df['Ano'] = df['Data'].dt.year
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
