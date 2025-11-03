import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Optional, Tuple, Dict

from cluster.cluster import criacao_variaveis_mes
from cluster.cluster_utils import preparar_dados as preparar_dados_original
from models.ClusterGraph import ClusterGraph


def preparar_features(df, usar_grafo: bool = False, grafo: Optional[ClusterGraph] = None):
    '''
    Preparar as features para o modelo de predição.
    Agora com suporte para features de grafo.
    '''
    df_copy = df.copy()
    df_copy = criacao_variaveis_mes(df_copy)
    
    if df_copy['Data'].dtype == 'object':
        df_copy['Data'] = pd.to_datetime(df_copy['Data'])
        
    df_copy['Ano'] = df_copy['Data'].dt.year
    df_copy['DiaAno'] = df_copy['Data'].dt.dayofyear
    
    # Adicionar features do grafo se solicitado
    if usar_grafo and grafo is not None:
        print('Adicionando features do grafo às features de predição...')
        df_copy = grafo.extrair_features_dataframe(df_copy)
    
    label_encoders = {}
    
    # Codificar todas as colunas categóricas (tipo object)
    colunas_categoricas = df_copy.select_dtypes(include=['object']).columns.tolist()
    colunas_remover = ['DataHora', 'Data']
    
    for col in colunas_categoricas:
        if col not in colunas_remover:
            le = LabelEncoder()
            df_copy[f'{col}_encoded'] = le.fit_transform(df_copy[col].astype(str))
            label_encoders[col] = le
            colunas_remover.append(col)
    
    # Remover colunas originais categóricas e de data
    df_copy = df_copy.drop(columns=[col for col in colunas_remover if col in df_copy.columns])
    
    return df_copy, label_encoders


def treinar_modelo(
    df,
    usar_grafo: bool = False,
    grafo: Optional[ClusterGraph] = None,
    mostrar_acuracia: bool = True,
    salvar_modelo: bool = False,
    caminho_modelo: str = 'modelo_cluster_grafo.pkl'
) -> Dict:
    '''
    Treinar o modelo de predição.
    Agora com suporte para features de grafo.
    
    Args:
        df: DataFrame com dados
        usar_grafo: Se True, usa features do grafo
        grafo: ClusterGraph construído (opcional)
        mostrar_acuracia: Se True, exibe métricas de acurácia
        salvar_modelo: Se True, salva o modelo em arquivo
        caminho_modelo: Caminho para salvar o modelo
    '''
    print(f'\n{"="*60}')
    print(f'TREINANDO MODELO DE PREDIÇÃO')
    if usar_grafo:
        print(f'Modo: COM features de grafo')
    else:
        print(f'Modo: SEM features de grafo (original)')
    print(f'{"="*60}\n')
    
    df_preparado, label_encoders = preparar_features(df, usar_grafo, grafo)
    
    seed = 4224
    
    y = df_preparado['RiscoFogo']
    X = df_preparado.drop(columns=['RiscoFogo'])
    
    print(f'Shape de X: {X.shape}')
    print(f'Features: {list(X.columns)}')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, shuffle=True)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f'\nTreinando RandomForestRegressor...')
    modelo = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=seed,
        n_jobs=-1
    )
    modelo.fit(X_train_scaled, y_train)
    
    metricas = {}
    
    if mostrar_acuracia:
        y_pred = modelo.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        margem = 10
        acertos_margem = np.abs(y_test - y_pred) <= margem
        acuracia_margem = acertos_margem.mean() * 100
        
        metricas = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'acuracia_margem_10': acuracia_margem
        }
        
        print(f'\n{"="*60}')
        print(f'MÉTRICAS DO MODELO')
        print(f'{"="*60}')
        print(f'R²: {r2:.4f} ({r2*100:.2f}%)')
        print(f'RMSE: {rmse:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'Acurácia (±{margem}): {acuracia_margem:.2f}%')
        print(f'{"="*60}\n')
        
        # Importância das features
        importancias = modelo.feature_importances_
        indices_ordenados = np.argsort(importancias)[::-1]
        
        print(f'Top 10 Features mais importantes:')
        for i in range(min(10, len(X.columns))):
            idx = indices_ordenados[i]
            print(f'  {i+1}. {X.columns[idx]}: {importancias[idx]:.4f}')
        
        # Se usar grafo, mostrar importância das features de grafo
        if usar_grafo:
            features_grafo = [col for col in X.columns if col.startswith('grafo_')]
            if features_grafo:
                print(f'\nImportância das features de grafo:')
                for feature in features_grafo:
                    idx = list(X.columns).index(feature)
                    print(f'  - {feature}: {importancias[idx]:.4f}')
    
    modelo_cluster = {
        'modelo': modelo,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': list(X.columns),
        'usar_grafo': usar_grafo,
        'grafo': grafo if usar_grafo else None,
        'metricas': metricas
    }
    
    if salvar_modelo:
        with open(caminho_modelo, 'wb') as f:
            pickle.dump(modelo_cluster, f)
        print(f'\n✅ Modelo salvo em "{caminho_modelo}"')
    
    return modelo_cluster


def comparar_modelos(df, grafo: ClusterGraph) -> Dict:
    '''
    Compara performance do modelo com e sem features de grafo.
    
    Args:
        df: DataFrame com dados
        grafo: ClusterGraph construído
        
    Returns:
        Dicionário com resultados da comparação
    '''

    print(f'\n{"="*60}')
    print(f'COMPARAÇÃO: MODELO COM vs SEM GRAFO')
    print(f'{"="*60}\n')

    # Treinar modelo sem grafo
    modelo_sem = treinar_modelo(df, usar_grafo=False, grafo=None, mostrar_acuracia=True)
    
    # Treinar modelo com grafo
    modelo_com = treinar_modelo(df, usar_grafo=True, grafo=grafo, mostrar_acuracia=True)
    
    # Comparar métricas
    metricas_sem = modelo_sem['metricas']
    metricas_com = modelo_com['metricas']
    
    resultados = {
        'r2_sem_grafo': metricas_sem['r2'],
        'r2_com_grafo': metricas_com['r2'],
        'melhoria_r2': metricas_com['r2'] - metricas_sem['r2'],
        'melhoria_r2_percentual': ((metricas_com['r2'] - metricas_sem['r2']) / metricas_sem['r2']) * 100,
        
        'rmse_sem_grafo': metricas_sem['rmse'],
        'rmse_com_grafo': metricas_com['rmse'],
        'melhoria_rmse': metricas_sem['rmse'] - metricas_com['rmse'],  # Menor é melhor
        'melhoria_rmse_percentual': ((metricas_sem['rmse'] - metricas_com['rmse']) / metricas_sem['rmse']) * 100,
        
        'mae_sem_grafo': metricas_sem['mae'],
        'mae_com_grafo': metricas_com['mae'],
        'melhoria_mae': metricas_sem['mae'] - metricas_com['mae'],  # Menor é melhor
        
        'acuracia_sem_grafo': metricas_sem['acuracia_margem_10'],
        'acuracia_com_grafo': metricas_com['acuracia_margem_10'],
        'melhoria_acuracia': metricas_com['acuracia_margem_10'] - metricas_sem['acuracia_margem_10'],
        
        'num_features_sem': len(modelo_sem['feature_names']),
        'num_features_com': len(modelo_com['feature_names'])
    }

    print(f'\n{"="*60}')
    print(f'RESUMO DA COMPARAÇÃO')
    print(f'{"="*60}')
    print(f'\nR² Score:')
    print(f'  - SEM Grafo: {metricas_sem["r2"]:.4f}')
    print(f'  - COM Grafo: {metricas_com["r2"]:.4f}')
    print(f'  - Melhoria: {resultados["melhoria_r2"]:+.4f} ({resultados["melhoria_r2_percentual"]:+.2f}%)')

    print(f'\nRMSE:')
    print(f'  - SEM Grafo: {metricas_sem["rmse"]:.4f}')
    print(f'  - COM Grafo: {metricas_com["rmse"]:.4f}')
    print(f'  - Melhoria: {resultados["melhoria_rmse"]:+.4f} ({resultados["melhoria_rmse_percentual"]:+.2f}%)')

    print(f'\nMAE:')
    print(f'  - SEM Grafo: {metricas_sem["mae"]:.4f}')
    print(f'  - COM Grafo: {metricas_com["mae"]:.4f}')
    print(f'  - Melhoria: {resultados["melhoria_mae"]:+.4f}')

    print(f'\nAcurácia (±10):')
    print(f'  - SEM Grafo: {metricas_sem["acuracia_margem_10"]:.2f}%')
    print(f'  - COM Grafo: {metricas_com["acuracia_margem_10"]:.2f}%')
    print(f'  - Melhoria: {resultados["melhoria_acuracia"]:+.2f}%')

    print(f'\nNúmero de Features:')
    print(f'  - SEM Grafo: {resultados["num_features_sem"]}')
    print(f'  - COM Grafo: {resultados["num_features_com"]}')
    print(f'{"="*60}\n')

    return resultados


def fazer_predicao(modelo_cluster: Dict, df_novos_dados: pd.DataFrame) -> np.ndarray:
    '''
    Faz predições usando o modelo treinado.
    '''
    
    usar_grafo = modelo_cluster.get('usar_grafo', False)
    grafo = modelo_cluster.get('grafo', None)
    
    # Preparar features
    df_preparado, _ = preparar_features(df_novos_dados, usar_grafo, grafo)
    
    # Remover RiscoFogo se existir
    if 'RiscoFogo' in df_preparado.columns:
        X = df_preparado.drop(columns=['RiscoFogo'])
    else:
        X = df_preparado
    
    # Garantir que as features estão na ordem correta
    X = X[modelo_cluster['feature_names']]
    
    # Normalizar
    X_scaled = modelo_cluster['scaler'].transform(X)
    
    # Predizer
    predicoes = modelo_cluster['modelo'].predict(X_scaled)
    
    return predicoes