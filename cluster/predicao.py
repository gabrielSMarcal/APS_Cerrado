import pandas as pd
from sklearn.preprocessing import LabelEncoder

from cluster.preparacao_dados import criar_variaveis_temporais

def preparar_features(df):
    '''
    Preparar as features para o modelo de predição
    '''
    
    df_copy = df.copy()
    df_copy = criar_variaveis_temporais(df_copy)
    
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
