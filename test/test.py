import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd                
import numpy as np                 
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score  
from sklearn.ensemble import RandomForestClassifier                                     
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix     
from sklearn.preprocessing import StandardScaler, LabelEncoder                          
from sklearn.pipeline import make_pipeline                                              
import seaborn as sns              
import matplotlib.pyplot as plt    
from data.check_data import check_errors  


# Carrega e limpeza inicial

cleaned_df, clean_list_dfs = check_errors() 
cleaned_df = cleaned_df.drop(columns=['Satelite', 'Pais', 'Bioma', 'DataHora', 'Estado', 'Municipio'], errors='ignore')

# Remove linhas com valores ausentes nas colunas principais
cleaned_df = cleaned_df.dropna(subset=['DiaSemChuva', 'Precipitacao', 'RiscoFogo', 'FRP', 'Latitude', 'Longitude'])

# Garante que a coluna RiscoFogo é numérica (converte texto para número e força erros como NaN)
cleaned_df['RiscoFogo'] = pd.to_numeric(cleaned_df['RiscoFogo'], errors='coerce')
cleaned_df = cleaned_df.dropna(subset=['RiscoFogo']) 

# Classificação do alvo
bins = [ -np.inf, 0.3, 0.7, np.inf ]   
labels = [0, 1, 2]                     

# Aplica o corte (classificação) nos valores de RiscoFogo
cleaned_df['RiscoFogo_class'] = pd.cut(cleaned_df['RiscoFogo'], bins=bins, labels=labels, include_lowest=True)


cleaned_df = cleaned_df.dropna(subset=['RiscoFogo_class']).copy()
cleaned_df['RiscoFogo_class'] = cleaned_df['RiscoFogo_class'].astype(int)

# Selecionar features numéricas (evitar incluir o alvo original)
num_cols = cleaned_df.select_dtypes(include='number').columns.tolist()  

# Remove o alvo original e a classe categórica (não podem ser usadas como preditores)
for col in ['RiscoFogo', 'RiscoFogo_class']:
    if col in num_cols:
        num_cols.remove(col)

# Cria o conjunto de entrada (X) e o alvo (y)
X = cleaned_df[num_cols].copy()                 
y = cleaned_df['RiscoFogo_class'].copy()        

# Garantir que X e y não têm NaNs (remove linhas com valores faltantes)
mascara = X.notna().all(axis=1) & y.notna()        
if mascara.sum() != len(y):                       
    print(f"Removendo {len(y) - mascara.sum()} linhas com NaNs em X ou y...")
X = X.loc[mascara].reset_index(drop=True) 
y = y.loc[mascara].reset_index(drop=True)

# Diagnósticos rápidos
print("Tamanho total após limpeza:", X.shape)   
print("Distribuição das classes (y):")          
print(y.value_counts())

# verifica vazamento de dados
for col in X.columns:
    if X[col].equals(y):
        print(f"⚠️ Vazamento detectado: coluna '{col}' é idêntica ao alvo 'y'!")

# Preparar split
Codifica_classe = LabelEncoder()                 
y_enc = Codifica_classe.fit_transform(y)         

# Mostra quantas amostras há por classe
amostra = pd.Series(y_enc).value_counts()
print("Contagem por classe (encoded):")
print(amostra)

# Garante que há pelo menos 2 amostras por classe antes de usar 'stratify'
use_stratify = True
if amostra.min() < 2:
    print("⚠️ Alguma classe tem menos de 2 amostras. Não vou usar stratify.")
    use_stratify = False

# Define o argumento do stratify
stratify_arg = y_enc if use_stratify else None

SEED = 20 
raw_treino_x, raw_teste_x, treino_y_enc, teste_y_enc = train_test_split(
    X, y_enc, test_size=0.2, random_state=SEED, stratify=stratify_arg
)

# Verifica sobreposição exata entre treino e teste
overlap = raw_treino_x.reset_index(drop=True).equals(raw_teste_x.reset_index(drop=True))
print("Sobreposição exata entre treino e teste?", overlap)


# Pipeline: scaler + RandomForest

# Cria uma sequência de etapas: padronizar dados -> treinar modelo
Mod_padronizacao = make_pipeline(
    StandardScaler(),                            # padroniza os dados (média 0, desvio 1)
    RandomForestClassifier(n_estimators=200, random_state=56, n_jobs=-1)  # modelo de floresta aleatória
)

# Treina o modelo com os dados de treino
Mod_padronizacao.fit(raw_treino_x, treino_y_enc)

# Faz previsões com o conjunto de teste, utiliza o raw_teste_x para evitar vazamento
previsoes_enc = Mod_padronizacao.predict(raw_teste_x)

# Decodifica as previsões (volta para as classes originais [o,1,2])
previsoes = Codifica_classe.inverse_transform(previsoes_enc)
teste_y = Codifica_classe.inverse_transform(teste_y_enc)

# Exibe métricas de desempenho
print('Acuracia: ', accuracy_score(teste_y, previsoes))
print("\nRelatório de classificação:\n", classification_report(teste_y, previsoes))

# Gera a matriz de confusão (visualização de acertos e erros)
cm = confusion_matrix(teste_y, previsoes)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão (conjunto de teste)')
plt.show()