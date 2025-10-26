import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

from data.connection import connection
from cluster_test.cluster import criacao_variaveis_mes

def preparar_features(df):
    
    '''
    Preparar as features para o modelo de predição
    '''
    
    df_copy = df.copy()
    df_copy = criacao_variaveis_mes(df_copy)
    
    if df_copy['Data'].dtype == 'object':
        df_copy['Data'] = pd.to_datetime(df_copy['Data'])
        
    df_copy['Ano'] = df_copy['Data'].dt.year
    df_copy['DiaAno'] = df_copy['Data'].dt.dayofyear
    
    label_encoders = {}
    
    if 'Estado' in df_copy.columns:
        le_estado = LabelEncoder()
        df_copy['Estado_encoded'] = le_estado.fit_transform(df_copy['Estado'].astype(str))
        label_encoders['Estado'] = le_estado

    if 'Municipio' in df_copy.columns:
        le_municipio = LabelEncoder()
        df_copy['Municipio_encoded'] = le_municipio.fit_transform(df_copy['Municipio'].astype(str))
        label_encoders['Municipio'] = le_municipio
        
    colunas_remover = ['DataHora', 'Data', 'Estado', 'Municipio']
    df_copy = df_copy.drop(columns=[col for col in colunas_remover if col in df_copy.columns])
    
    return df_copy, label_encoders

def treinar_modelo(df, mostrar_acuracia=False):
    
    '''
    Treinar o modelo de predição
    '''
    
    df_preparado, label_encoders = preparar_features(df)
    
    seed = 4224
    
    y = df_preparado['RiscoFogo']
    X = df_preparado.drop(columns=['RiscoFogo'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, shuffle=True
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    modelo = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=seed,
        n_jobs=-1
    )
    modelo.fit(X_train_scaled, y_train)
    
    # if mostrar_acuracia:
    #     y_pred = modelo.predict(X_test_scaled)
    #     r2 = r2_score(y_test, y_pred)
    #     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    #     mae = mean_absolute_error(y_test, y_pred)
    #     margem = 10
    #     acertos_margem = np.abs(y_test - y_pred) <= margem
    #     acuracia_margem = acertos_margem.mean() * 100
        
    #     print(f'R²: {r2:.4f} ({r2*100:.2f}%)')
    #     print(f'RMSE: {rmse:.4f}')
    #     print(f'MAE: {mae:.4f}')
    #     print(f'Acurácia (±{margem}): {acuracia_margem:.2f}%')
    
    # R²: 0.8959 (89.59%) | RMSE: 10.3698 | MAE: 4.5224 | Acurácia (±10): 77.96%
    
    # Salvar tudo em um único arquivo
    modelo_cluster = {
        'modelo': modelo,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': list(X.columns)
    }
    
    # with open('modelo_cluster.pkl', 'wb') as f:
    #     pickle.dump(modelo_cluster, f)
    
    # print('Modelo treinado e salvo com sucesso!')
    
    return modelo_cluster

df = connection()
# treinar_modelo(df, mostrar_acuracia=True)  # Para testar
treinar_modelo(df)  # Para treinar