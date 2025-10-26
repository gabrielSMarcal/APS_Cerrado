import pandas as pd
from sklearn.calibration import LabelEncoder
from cluster_test.cluster import criacao_variaveis_mes


def preparar_dados(df, modelo_cluster=None):
    """
    Prepara os dados para o modelo, incluindo criação de variáveis de mês, ano e codificação de categorias.
    Se um modelo for fornecido, aplica os LabelEncoders do modelo.
    """
    df_copy = df.copy()
    
    # Criar variáveis dummy para os meses
    df_copy = criacao_variaveis_mes(df_copy)
    
    # Garantir que a coluna 'Data' esteja no formato datetime
    if df_copy['Data'].dtype == 'object':
        df_copy['Data'] = pd.to_datetime(df_copy['Data'], errors='coerce')
    
    # Adicionar colunas de ano e dia do ano
    df_copy['Ano'] = df_copy['Data'].dt.year
    df_copy['DiaAno'] = df_copy['Data'].dt.dayofyear
    
    # Codificar colunas categóricas, se existirem
    if modelo_cluster and 'label_encoders' in modelo_cluster:
        label_encoders = modelo_cluster['label_encoders']
        if 'Estado' in df_copy.columns and 'Estado' in label_encoders:
            df_copy['Estado_encoded'] = label_encoders['Estado'].transform(df_copy['Estado'].astype(str))
        if 'Municipio' in df_copy.columns and 'Municipio' in label_encoders:
            df_copy['Municipio_encoded'] = label_encoders['Municipio'].transform(df_copy['Municipio'].astype(str))
    else:
        # Criar novos LabelEncoders, se necessário
        label_encoders = {}
        if 'Estado' in df_copy.columns:
            le_estado = LabelEncoder()
            df_copy['Estado_encoded'] = le_estado.fit_transform(df_copy['Estado'].astype(str))
            label_encoders['Estado'] = le_estado
        if 'Municipio' in df_copy.columns:
            le_municipio = LabelEncoder()
            df_copy['Municipio_encoded'] = le_municipio.fit_transform(df_copy['Municipio'].astype(str))
            label_encoders['Municipio'] = le_municipio
    
    # Remover colunas desnecessárias
    colunas_remover = ['DataHora', 'Data', 'Estado', 'Municipio']
    df_copy = df_copy.drop(columns=[col for col in colunas_remover if col in df_copy.columns])
    
    return df_copy, label_encoders if not modelo_cluster else df_copy