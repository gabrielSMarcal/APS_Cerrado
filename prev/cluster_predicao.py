import pandas as pd
import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict
import multiprocessing

from models.TAD.ClusterGraph import ClusterGraph
from cluster.preparacao_dados import preparar_para_predicao, validar_features
from data.connection import connection
from render.visualizacoes_modelo import gerar_todas_visualizacoes


def treinar_modelo(
    df,
    usar_grafo: bool = False,
    grafo: Optional[ClusterGraph] = None,
    mostrar_acuracia: bool = True,
    salvar_modelo: bool = False,
    caminho_modelo: str = 'modelo_random_forest.pkl'
) -> Dict:
    """
    Treina modelo Random Forest para predição de risco de fogo.
    
    OTIMIZAÇÕES:
    - NÃO salva o grafo completo (apenas features extraídas)
    - Usa preparação centralizada
    - Modelo leve (~5MB ao invés de 1.4GB)
    """
    inicio = time.time()
    
    print(f'\n{"="*60}')
    print(f'TREINANDO MODELO DE PREDICAO')
    if usar_grafo:
        print(f'Modo: COM features de grafo')
    else:
        print(f'Modo: SEM features de grafo (básico)')
    print(f'{"="*60}\n')
    
    # ✅ USAR PREPARAÇÃO CENTRALIZADA
    df_preparado, label_encoders = preparar_para_predicao(
        df, 
        usar_grafo=usar_grafo, 
        grafo=grafo
    )
    
    seed = 4224
    
    # Separar features e target
    y = df_preparado['RiscoFogo']
    X = df_preparado.drop(columns=['RiscoFogo'])
    
    print(f'Shape de X: {X.shape}')
    print(f'Features utilizadas ({len(X.columns)}): {list(X.columns)}')
    
    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, shuffle=True
    )
    
    # Otimização: usar float32 ao invés de float64
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Treinar Random Forest
    print(f'\nTreinando RandomForestRegressor OTIMIZADO...')
    num_cores = multiprocessing.cpu_count()
    print(f'Usando {num_cores} cores de CPU')
    print(f'Par\u00e2metros otimizados:')
    print(f'  - n_estimators: 70 (equil\u00edbrio tamanho/acur\u00e1cia)')
    print(f'  - max_depth: 12 (12 meses do ano)')
    print(f'  - min_samples_leaf: 5 (robusto)')
    
    modelo = RandomForestRegressor(
        n_estimators=70,
        max_depth=12,
        min_samples_split=6,
        max_features='sqrt',
        min_samples_leaf=5,
        bootstrap=True,
        oob_score=True,
        random_state=seed,
        n_jobs=num_cores,
        verbose=1
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
            'acuracia_margem_10': acuracia_margem,
            'oob_score': modelo.oob_score_
        }
        
        print(f'\n{"="*60}')
        print(f'METRICAS DO MODELO')
        print(f'{"="*60}')
        print(f'R2: {r2:.4f} ({r2*100:.2f}%)')
        print(f'OOB Score: {modelo.oob_score_:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'Acuracia (+-{margem}): {acuracia_margem:.2f}%')
        print(f'{"="*60}\n')
        
        # Importância das features (top 10)
        importancias = modelo.feature_importances_
        indices_ordenados = np.argsort(importancias)[::-1][:10]
        
        print(f'Top 10 Features mais importantes:')
        for i, idx in enumerate(indices_ordenados, 1):
            print(f'  {i}. {X.columns[idx]}: {importancias[idx]:.4f}')
        
        # Gerar visualizações
        print('\n' + '='*60)
        print('GERANDO VISUALIZAÇÕES')
        print('='*60)

        try:
            df_test_viz = None
            if hasattr(X_test, 'index'):
                indices_test = X_test.index
                if 'df' in locals() or 'df' in globals():
                    df_test_viz = df.loc[indices_test].copy()
            
            gerar_todas_visualizacoes(
                modelo=modelo,
                X=X_train,
                y_test=y_test,
                y_pred=y_pred,
                df_test=df_test_viz
            )
            
            print('✅ Visualizações geradas com sucesso!')
            
        except Exception as e:
            print(f'⚠️ Erro ao gerar visualizações: {e}')
            import traceback
            traceback.print_exc()

        print('='*60 + '\n')
        
        # Mostrar importância das features de grafo
        if usar_grafo:
            features_grafo = [col for col in X.columns if col.startswith('grafo_')]
            if features_grafo:
                print(f'\nImportancia das features de grafo:')
                for feature in features_grafo:
                    idx = list(X.columns).index(feature)
                    print(f'  - {feature}: {importancias[idx]:.4f}')
    
    # ✅ MODELO LEVE: NÃO SALVAR O GRAFO!
    modelo_cluster = {
        'modelo': modelo,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': list(X.columns),
        'usar_grafo': usar_grafo,
        'metricas': metricas
    }
    
    if salvar_modelo:
        with open(caminho_modelo, 'wb') as f:
            pickle.dump(modelo_cluster, f)
        
        # Verificar tamanho do arquivo
        import os
        tamanho_mb = os.path.getsize(caminho_modelo) / (1024 * 1024)
        print(f'\nModelo salvo em "{caminho_modelo}"')
        print(f'Tamanho do arquivo: {tamanho_mb:.2f} MB')
        
        if tamanho_mb > 100:
            print(f'⚠️ AVISO: Modelo muito grande! Deveria ter ~5-10MB')
    
    tempo_total = time.time() - inicio
    print(f'\nTempo de execucao: {tempo_total:.2f} segundos')
    
    return modelo_cluster


def fazer_predicao(modelo_cluster: Dict, df_novos_dados: pd.DataFrame) -> np.ndarray:
    """
    Faz predições usando modelo treinado.
    
    IMPORTANTE: Se o modelo foi treinado COM grafo, você precisa 
    passar um grafo construído externamente.
    """
    inicio = time.time()
    
    usar_grafo = modelo_cluster.get('usar_grafo', False)
    label_encoders = modelo_cluster.get('label_encoders', None)
    
    print('\nFazendo predicoes...')
    
    # ✅ PREPARAR DADOS (sem grafo, pois não foi salvo)
    df_preparado, _ = preparar_para_predicao(
        df_novos_dados, 
        usar_grafo=False,  # Grafo não está disponível
        grafo=None,
        label_encoders=label_encoders
    )
    
    # Remover RiscoFogo se existir
    if 'RiscoFogo' in df_preparado.columns:
        X = df_preparado.drop(columns=['RiscoFogo'])
    else:
        X = df_preparado
    
    # Validar e reordenar features
    X = validar_features(X, modelo_cluster['feature_names'])
    
    # Otimização: usar float32
    X = X.astype(np.float32)
    
    # Normalizar e predizer
    X_scaled = modelo_cluster['scaler'].transform(X)
    predicoes = modelo_cluster['modelo'].predict(X_scaled)
    
    tempo_total = time.time() - inicio
    print(f'Predicoes concluidas em {tempo_total:.2f} segundos')
    print(f'Total de predicoes: {len(predicoes)}')
    
    return predicoes