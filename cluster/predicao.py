import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

from data.connection import connection
from cluster.cluster import criacao_variaveis_mes

def preparar_features(df):
    
    '''
    Preparar as features para o modelo de predição
    '''
    
    df_copy = df.copy()
    df_copy = criacao_variaveis_mes(df_copy)
    
    if df_copy['Data'].dtype == 'object':
        df_copy['Data'] = pd.to_datetime(df_copy['Data'])
        
    df_copy['Ano'] = df_copy['Data'].dt.year
    df_copy['Dia'] = df_copy['Data'].dt.day
    df_copy['DiaSemana'] = df_copy['Data'].dt.dayofweek
    df_copy = ['DiaAno'] = df_copy['Data'].dt.dayofyear
    
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
