import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Tuple, Dict
from models.TAD.ClusterGraph import ClusterGraph


def criar_variaveis_temporais(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Cria variáveis temporais (mês, ano, dia) a partir da coluna de data.
    Substitui criacao_variaveis_mes() com nome mais descritivo.
    '''
    
    df_copy = df.copy()
    
    # Identificar coluna de data
    if 'DataHora' in df_copy.columns:
        df_copy['Data'] = pd.to_datetime(df_copy['DataHora'], errors='coerce')
    elif 'Data' in df_copy.columns:
        df_copy['Data'] = pd.to_datetime(df_copy['Data'], errors='coerce')
    else:
        raise KeyError('Nenhuma coluna de data encontrada ("DataHora" ou "Data").')
    
    # Verificar valores inválidos
    if df_copy['Data'].isna().any():
        raise ValueError('A coluna "Data" contém valores inválidos após conversão.')
    
    # Extrair características temporais
    df_copy['Mes'] = df_copy['Data'].dt.month
    df_copy['Ano'] = df_copy['Data'].dt.year
    df_copy['DiaAno'] = df_copy['Data'].dt.dayofyear
    df_copy['DiaSemana'] = df_copy['Data'].dt.dayofweek
    
    # Criar variáveis dummy para meses
    for mes in range(1, 13):
        df_copy[f'Mes_{mes}'] = (df_copy['Mes'] == mes).astype(int)
    
    df_copy = df_copy.drop(columns=['Mes'])
    
    return df_copy


def codificar_categoricas(
    df: pd.DataFrame,
    label_encoders: Optional[Dict[str, LabelEncoder]] = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    '''
    Codifica colunas categóricas usando LabelEncoder.
    '''
    
    df_copy = df.copy()
    
    if label_encoders is None:
        label_encoders = {}
        criar_novos = True
    else:
        criar_novos = False
    
    # Identificar colunas categóricas
    colunas_categoricas = df_copy.select_dtypes(include=['object']).columns.tolist()
    colunas_excluir = ['DataHora', 'Data']
    
    for col in colunas_categoricas:
        if col in colunas_excluir:
            continue
            
        if criar_novos:
            le = LabelEncoder()
            df_copy[f'{col}_encoded'] = le.fit_transform(df_copy[col].astype(str))
            label_encoders[col] = le
        else:
            if col in label_encoders:
                df_copy[f'{col}_encoded'] = label_encoders[col].transform(df_copy[col].astype(str))
    
    # Remover colunas originais
    colunas_remover = colunas_categoricas + ['DataHora', 'Data']
    df_copy = df_copy.drop(columns=[c for c in colunas_remover if c in df_copy.columns])
    
    return df_copy, label_encoders


def preparar_para_clustering(
    df: pd.DataFrame,
    usar_grafo: bool = False,
    grafo: Optional[ClusterGraph] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Dict[str, LabelEncoder], list]:
    '''
    Prepara dados especificamente para clustering.
    '''
    
    # Criar variáveis temporais
    df_prep = criar_variaveis_temporais(df)
    
    # Adicionar features do grafo se solicitado
    if usar_grafo and grafo is not None:
        print('Adicionando features do grafo aos dados de clustering...')
        df_prep = grafo.extrair_features_dataframe(df_prep)
    
    # Codificar categóricas
    df_prep, label_encoders = codificar_categoricas(df_prep)
    
    # Separar features e target
    if 'RiscoFogo' in df_prep.columns:
        y = df_prep['RiscoFogo']
        X = df_prep.drop(columns=['RiscoFogo'])
    else:
        y = None
        X = df_prep
    
    # Normalização
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X, X_scaled, y, label_encoders, list(X.columns)


def preparar_para_predicao(
    df: pd.DataFrame,
    usar_grafo: bool = False,
    grafo: Optional[ClusterGraph] = None,
    label_encoders: Optional[Dict[str, LabelEncoder]] = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    '''
    Prepara dados especificamente para predição.
    '''
    
    # Criar variáveis temporais
    df_prep = criar_variaveis_temporais(df)
    
    # Adicionar features do grafo se solicitado
    if usar_grafo and grafo is not None:
        print('Adicionando features do grafo aos dados de predição...')
        df_prep = grafo.extrair_features_dataframe(df_prep)
    
    # Codificar categóricas
    df_prep, encoders = codificar_categoricas(df_prep, label_encoders)
    
    return df_prep, encoders


def validar_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    '''
    Valida e reordena features para garantir compatibilidade com modelo treinado.
    '''
    
    # Adicionar features faltantes com zeros
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Reordenar para match exato com modelo
    df = df[feature_names]
    
    return df