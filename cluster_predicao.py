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


def treinar_modelo(
    df,
    usar_grafo: bool = False,
    grafo: Optional[ClusterGraph] = None,
    mostrar_acuracia: bool = True,
    salvar_modelo: bool = False,
    caminho_modelo: str = 'modelo_random_forest.pkl'
) -> Dict:
    
    inicio = time.time()
    
    print(f'\n{"="*60}')
    print(f'TREINANDO MODELO DE PREDICAO OTIMIZADO')
    if usar_grafo:
        print(f'Modo: COM features de grafo')
    else:
        print(f'Modo: SEM features de grafo (básico)')
    print(f'{"="*60}\n')
    
    df_preparado, label_encoders = preparar_para_predicao(
        df, 
        usar_grafo=usar_grafo, 
        grafo=grafo
    )
    
    seed = 4224
    
    y = df_preparado['RiscoFogo']
    X = df_preparado.drop(columns=['RiscoFogo'])
    
    print(f'Shape de X: {X.shape}')
    print(f'Features utilizadas ({len(X.columns)}): {list(X.columns)}')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, shuffle=True
    )
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f'\nTreinando RandomForestRegressor OTIMIZADO...')
    num_cores = multiprocessing.cpu_count()
    print(f'Usando {num_cores} cores de CPU')
    print(f'Parâmetros otimizados para modelo leve (<100MB):')
    print(f'  - n_estimators: 40 (antes: 100)')
    print(f'  - max_depth: 12 (antes: 20)')
    print(f'  - min_samples_leaf: 8 (antes: 2)')
    
    modelo = RandomForestRegressor(
        n_estimators=40,
        max_depth=12,
        min_samples_split=8,
        max_features='sqrt',
        min_samples_leaf=8,
        bootstrap=True,
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
            'acuracia_margem_10': acuracia_margem
        }
        
        print(f'\n{"="*60}')
        print(f'METRICAS DO MODELO')
        print(f'{"="*60}')
        print(f'R2: {r2:.4f} ({r2*100:.2f}%)')
        print(f'RMSE: {rmse:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'Acuracia (+-{margem}): {acuracia_margem:.2f}%')
        print(f'{"="*60}\n')
        
        importancias = modelo.feature_importances_
        indices_ordenados = np.argsort(importancias)[::-1][:10]
        
        print(f'Top 10 Features mais importantes:')
        for i, idx in enumerate(indices_ordenados, 1):
            print(f'  {i}. {X.columns[idx]}: {importancias[idx]:.4f}')
        
        if usar_grafo:
            features_grafo = [col for col in X.columns if col.startswith('grafo_')]
            if features_grafo:
                print(f'\nImportancia das features de grafo:')
                for feature in features_grafo:
                    idx = list(X.columns).index(feature)
                    print(f'  - {feature}: {importancias[idx]:.4f}')
    
    modelo_cluster = {
        'modelo': modelo,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': list(X.columns),
        'usar_grafo': usar_grafo,
        'metricas': metricas
    }
    
    if salvar_modelo:
        print(f'\nSalvando modelo otimizado...')
        with open(caminho_modelo, 'wb') as f:
            pickle.dump(modelo_cluster, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        import os
        tamanho_mb = os.path.getsize(caminho_modelo) / (1024 * 1024)
        print(f'Modelo salvo em "{caminho_modelo}"')
        print(f'Tamanho do arquivo: {tamanho_mb:.2f} MB')
        
        if tamanho_mb > 100:
            print(f'⚠️ AVISO: Modelo ainda grande ({tamanho_mb:.1f}MB)!')
            print(f'Considere reduzir mais: n_estimators=20, max_depth=8')
        elif tamanho_mb < 50:
            print(f'✅ Modelo otimizado com sucesso! (<50MB)')
        else:
            print(f'✅ Modelo dentro do limite do GitHub (<100MB)')
    
    tempo_total = time.time() - inicio
    print(f'\nTempo de execucao: {tempo_total:.2f} segundos')
    
    return modelo_cluster


def fazer_predicao(modelo_cluster: Dict, df_novos_dados: pd.DataFrame) -> np.ndarray:
    
    inicio = time.time()
    
    usar_grafo = modelo_cluster.get('usar_grafo', False)
    label_encoders = modelo_cluster.get('label_encoders', None)
    
    print('\nFazendo predicoes...')
    
    df_preparado, _ = preparar_para_predicao(
        df_novos_dados, 
        usar_grafo=False,
        grafo=None,
        label_encoders=label_encoders
    )
    
    if 'RiscoFogo' in df_preparado.columns:
        X = df_preparado.drop(columns=['RiscoFogo'])
    else:
        X = df_preparado
    
    X = validar_features(X, modelo_cluster['feature_names'])
    X = X.astype(np.float32)
    
    X_scaled = modelo_cluster['scaler'].transform(X)
    predicoes = modelo_cluster['modelo'].predict(X_scaled)
    
    tempo_total = time.time() - inicio
    print(f'Predicoes concluidas em {tempo_total:.2f} segundos')
    print(f'Total de predicoes: {len(predicoes)}')
    
    return predicoes


if __name__ == "__main__":
    inicio_programa = time.time()
    
    print("="*60)
    print("SISTEMA DE PREDICAO DE RISCO DE FOGO")
    print("="*60)
    
    print("\nCarregando dados...")
    try:
        df = connection()
        
        if df is None or df.empty:
            print("Erro: Nenhum dado foi carregado")
            exit(1)
            
        print(f"Dados carregados: {len(df)} registros")
        print(f"Colunas: {list(df.columns)}")
        
        print("\n[OPÇÃO 1] Treinando modelo SEM grafo (básico)...")
        modelo_basico = treinar_modelo(
            df,
            usar_grafo=False,
            grafo=None,
            mostrar_acuracia=True,
            salvar_modelo=True,
            caminho_modelo='./models/modelo_basico.pkl'
        )
        
        resposta = input("\nDeseja treinar modelo COM grafo? (s/n): ")
        
        if resposta.lower() == 's':
            print("\n[OPÇÃO 2] Construindo grafo...")
            grafo = ClusterGraph()
            
            tempo_grafo = time.time()
            grafo.construir_grafo_dataframe(
                df,
                threshold_km=50.0,
                threshold_dias=7,
                usar_temporal=True,
                usar_espacial=True,
                max_conexoes_por_vertice=10,
                grid_size_km=50.0,
                janela_temporal_dias=14,
                mostrar_progresso=True
            )
            print(f"Grafo construído em {time.time() - tempo_grafo:.2f}s")
            print(f"Grafo: {len(grafo.get_pontos())} vértices")
            
            print("\nTreinando modelo COM grafo...")
            modelo_grafo = treinar_modelo(
                df,
                usar_grafo=True,
                grafo=grafo,
                mostrar_acuracia=True,
                salvar_modelo=True,
                caminho_modelo='./models/modelo_com_grafo.pkl'
            )
            
            print("\n✅ Modelos treinados com sucesso!")
            print(f"- Modelo básico: ./models/modelo_basico.pkl")
            print(f"- Modelo com grafo: ./models/modelo_com_grafo.pkl")
        else:
            print("\n✅ Modelo básico treinado com sucesso!")
            print(f"- Modelo salvo: ./models/modelo_basico.pkl")
        
    except ImportError as e:
        print(f"Erro ao importar módulo: {e}")
        print("Certifique-se de que o módulo data.connection está disponível")
    except Exception as e:
        print(f"Erro ao executar: {e}")
        import traceback
        traceback.print_exc()
    
    tempo_total_programa = time.time() - inicio_programa
    print(f"\n{'='*60}")
    print(f"TEMPO TOTAL DE EXECUÇÃO: {tempo_total_programa:.2f} segundos")
    print(f"TEMPO EM MINUTOS: {tempo_total_programa/60:.2f} minutos")
    print(f"{'='*60}")
