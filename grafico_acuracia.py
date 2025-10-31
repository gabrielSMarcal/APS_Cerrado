# avaliar_e_graficos.py
"""
Script para avaliar modelo e gerar gráficos PNG (matplotlib) e mapas (folium).
Coloque este arquivo na raiz do seu projeto e rode: python avaliar_e_graficos.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# imports do seu projeto: ajustar se os caminhos forem diferentes
from cluster.cluster_utils import preparar_dados
from cluster.cluster_predicao import treinar_modelo  # necessário se modelo não existir
from data.connection import connection  # deve retornar DataFrame com colunas Data, Latitude, Longitude, Estado, Municipio, RiscoFogo

import folium
from folium.plugins import MarkerCluster, HeatMap

# ----- Configurações -----
OUT_DIR = './avaliacao_outputs'
MODEL_PATH = './models/modelo_cluster.pkl'   # ajuste se necessário
MARGEM = 10           # margem ± para considerar 'acerto'
SAMPLE_MAP = None     # se quiser amostrar mapas (ex: 5000), ou None para todos
# --------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Funções ----------------

def carregar_modelo_ou_treinar(model_path=MODEL_PATH):
    """
    Tenta carregar modelo de model_path. Se não existir, treina e salva.
    """
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            modelo_cluster = pickle.load(f)
        print(f"Modelo carregado de {model_path}")
        return modelo_cluster

    print(f"Arquivo {model_path} não encontrado. Treinando modelo automaticamente...")
    df_hist = connection()
    if df_hist is None or len(df_hist) == 0:
        raise RuntimeError("Dados históricos vazios — não é possível treinar o modelo automaticamente.")

    modelo_cluster = treinar_modelo(df_hist)

    # garantir pasta
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(modelo_cluster, f)
    print(f"Modelo treinado e salvo em {model_path}")

    return modelo_cluster


def preparar_e_prever_tudo(df_raw, modelo_cluster):
    df = df_raw.copy()
    df['Data'] = pd.to_datetime(df['Data'])
    df['Ano'] = df['Data'].dt.year

    df_preparado, extras = preparar_dados(df, modelo_cluster=modelo_cluster)

    # copiar colunas úteis
    keys_to_copy = ['Data', 'Latitude', 'Longitude', 'Estado', 'Municipio', 'RiscoFogo', 'Ano']
    df_preparado = df_preparado.reset_index(drop=True)
    df = df.reset_index(drop=True)
    for k in keys_to_copy:
        if k in df.columns and k not in df_preparado.columns:
            df_preparado[k] = df[k]

    feature_names = modelo_cluster.get('feature_names', [c for c in df_preparado.columns if c != 'RiscoFogo'])
    features_disponiveis = [f for f in feature_names if f in df_preparado.columns]
    if len(features_disponiveis) == 0:
        raise RuntimeError("Nenhuma feature disponível após preparar_dados.")

    X_all = df_preparado[features_disponiveis]

    scaler = modelo_cluster.get('scaler', None)
    if scaler is not None:
        X_all_scaled = scaler.transform(X_all)
    else:
        X_all_scaled = X_all.values

    model = modelo_cluster.get('modelo')
    if model is None:
        raise RuntimeError("modelo_cluster não contém chave 'modelo'.")

    y_pred = model.predict(X_all_scaled)
    y_pred = np.clip(y_pred, 0, 100).round().astype(int)
    df_preparado['Risco_pred'] = y_pred

    return df_preparado, features_disponiveis


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


def gerar_mapas_2025(df_eval, out_dir=OUT_DIR, sample=None):
    """
    Gera mapa Folium apenas para o ano de 2025 mostrando:
      - ACERTO (verde)
      - SOBRE-estimativa (laranja)
      - SUB-estimativa (vermelho)
    Salva mapa_2025_acertos_erros.html
    """
    df_2025 = df_eval[df_eval['Ano'] == 2025].copy()
    if df_2025.empty:
        print("Não há dados para o ano de 2025.")
        return None

    if sample is not None and len(df_2025) > sample:
        df_2025 = df_2025.sample(sample, random_state=42)

    # Classificação acerto/sobre/sub
    def resultado_margem(row):
        true = float(row['RiscoFogo'])
        pred = float(row['Risco_pred'])
        if abs(true - pred) <= MARGEM:
            return 'ACERTO'
        return 'SOBRE' if pred > true else 'SUB'

    df_2025['result_margem'] = df_2025.apply(resultado_margem, axis=1)

    # Criar mapa
    start_coords = [df_2025['Latitude'].median(), df_2025['Longitude'].median()]
    m = folium.Map(location=start_coords, zoom_start=5, tiles='CartoDB Positron')
    mc = MarkerCluster().add_to(m)
    colors = {'ACERTO':'green', 'SOBRE':'orange', 'SUB':'red'}

    for _, r in df_2025.iterrows():
        popup = (f"Ano:{int(r['Ano'])}<br>"
                 f"Data:{pd.to_datetime(r['Data']).date()}<br>"
                 f"Risco_true:{int(r['RiscoFogo'])} - Risco_pred:{int(r['Risco_pred'])}<br>"
                 f"Resultado:{r['result_margem']}")
        folium.CircleMarker(
            location=(r['Latitude'], r['Longitude']),
            radius=4,
            color=colors.get(r['result_margem'], 'gray'),
            fill=True,
            fill_opacity=0.7,
            popup=popup
        ).add_to(mc)

    map_path = os.path.join(out_dir, 'map_2025_acertos_erros.html')
    m.save(map_path)
    print("Mapa 2025 salvo:", map_path)
    return map_path

def gerar_heatmap_erro(df_eval, out_dir=OUT_DIR):
    heat_points = df_eval[['Latitude', 'Longitude', 'RiscoFogo', 'Risco_pred']].dropna().copy()
    heat_points['erro_abs'] = (heat_points['RiscoFogo'] - heat_points['Risco_pred']).abs()

    if heat_points.empty:
        return None

    max_err = heat_points['erro_abs'].max()
    weights = (heat_points['erro_abs'] / max_err).tolist() if max_err > 0 else [0.0]*len(heat_points)
    heat_data = [[row['Latitude'], row['Longitude'], w] for (_, row), w in zip(heat_points.iterrows(), weights)]
    center = [heat_points['Latitude'].median(), heat_points['Longitude'].median()]

    m_heat = folium.Map(location=center, zoom_start=5, tiles='CartoDB Positron')
    HeatMap(heat_data, radius=10, blur=12, max_zoom=6).add_to(m_heat)

    heat_path = os.path.join(out_dir, 'heatmap_erro_agregado.html')
    m_heat.save(heat_path)
    return heat_path

# ---------------- Execução principal ----------------

if __name__ == '__main__':
    print("Iniciando avaliação e geração de gráficos...")

    modelo_cluster = carregar_modelo_ou_treinar(MODEL_PATH)
    df_hist = connection()
    print("Dados carregados. Linhas:", len(df_hist))

    df_eval, features_used = preparar_e_prever_tudo(df_hist, modelo_cluster)
    print("Preparação + previsão concluída. Features usadas:", len(features_used))

    metrics_df, margin_df = calcular_metricas_por_ano(df_eval, y_col='RiscoFogo')
    print("Métricas por ano calculadas:")
    print(metrics_df)

    plot_e_salvar_metricas(metrics_df, margin_df, out_dir=OUT_DIR)

    # Gerar apenas mapa de 2025
    map_2025 = gerar_mapas_2025(df_eval, out_dir=OUT_DIR, sample=SAMPLE_MAP)

    heat_path = gerar_heatmap_erro(df_eval, out_dir=OUT_DIR)

    print("\nProcesso finalizado. Arquivos em:", OUT_DIR)
    print("Mapa 2025 gerado:", map_2025)
    print("Heatmap:", heat_path)
