import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder

from data import connection as c

df = c.connection()

# Converter para data
df['Data'] = pd.to_datetime(df['Data'])

# Extrair o mês
df['Mes'] = df['Data'].dt.month

for mes in range (1, 13):
    df[f'Mes_{mes}'] = (df['Mes'] == mes).astype(int)
        
# Remover o mês
df = df.drop(columns=['Mes'])

