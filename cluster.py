import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_samples

from sklearn.metrics import silhouette_score

from data import connection as c

df = pd.read_csv('db_2024.csv')

df['Data'] = pd.to_datetime(df['DataHora']).dt.date

def criacao_variaveis_mes(df):

    # Converter para data
    df['Data'] = pd.to_datetime(df['Data'])

    # Extrair o mês
    df['Mes'] = df['Data'].dt.month

    # Criação dos meses dummy
    for mes in range (1, 13):
        df[f'Mes_{mes}'] = (df['Mes'] == mes).astype(int)
            
    # Remover o mês
    df = df.drop(columns=['Mes'])
    
    return df

df = criacao_variaveis_mes(df)

def preparar_dados(df):
    # Criar uma cópia para não modificar o original
    df_copy = df.copy()
    
    # Remover colunas não necessárias para o modelo
    colunas_remover = ['Data']
    
    # Verificar se existem outras colunas de data/objeto
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            if col not in colunas_remover:
                colunas_remover.append(col)
    
    df_copy = df_copy.drop(columns=colunas_remover)
    
    # Separar features e target
    y = df_copy['RiscoFogo']
    X = df_copy.drop(columns=['RiscoFogo'])
    
    # # Verificar se há valores não numéricos
    # print(f"Colunas em X: {X.columns.tolist()}")
    # print(f"Tipos de dados em X:\n{X.dtypes}")
    
    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, X_scaled, y, scaler

def encontrar_cluster(X_scaled, k_range=range(2, 20)):
    
    inercias = []
    silhuetas = []
    
    print(f'Linhas em X_scaled: {X_scaled.shape[0]}')
    
    for k in k_range:
        print(f"Calculando para k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=657, n_init='auto')
        kmeans.fit(X_scaled)
        inercias.append(kmeans.inertia_)
        silhuetas.append(silhouette_score(X_scaled, kmeans.labels_))
        print(f" - Inércia: {kmeans.inertia_:.2f}, Silhouette Score: {silhuetas[-1]:.4f}")
        
        
    # Resultados
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Método do cotovelo
    ax1.plot(k_range, inercias, 'bo-')
    ax1.axvline(x=12, color='r', linestyle='--', label='12 Meses')
    ax1.set_xlabel('Número de Clusters (k)')
    ax1.set_ylabel('Inércia')
    ax1.set_title('Método do Cotovelo')
    ax1.legend()
    ax1.grid(True)
    
    # Método da silhueta
    ax2.plot(k_range, silhuetas, 'go-')
    ax2.axvline(x=12, color='r', linestyle='--', label='12 Meses')
    ax2.set_xlabel('Número de Clusters (k)')
    ax2.set_ylabel('Score de Silhueta')
    ax2.set_title('Método da Silhueta')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # SALVAR O GRÁFICO COMO IMAGEM
    plt.savefig('graficos_clustering.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico salvo em 'graficos_clustering.png'")
    
    plt.show()
    
    melhor_k = k_range[np.argmax(silhuetas)]
    print(f'Melhor número de clusters (k) baseado na silhueta: {melhor_k}')
    print(f'Silhouette Score em k=12 {silhuetas[10]:.4f}')
    
    # RETORNAR TAMBÉM OS DADOS PARA RECRIAR O GRÁFICO NO DASH
    return melhor_k, silhuetas, inercias, k_range

def criar_clusters(X_scaled, clusters, y):
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(15, 5))
    
    # Plot dos clusters
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Visualização dos Clusters (PCA)')
    
    # Plot dos riscos de fogo
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='YlOrRd', alpha=0.6)
    plt.colorbar(scatter, label='Risco de Fogo')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Risco de Fogo (PCA)')
    
    plt.tight_layout()
    plt.show()
    


# Executar o pipeline de clustering
if __name__ == "__main__":
    print("Preparando dados...")
    X, X_scaled, y, scaler = preparar_dados(df)
    
    print("\nEncontrando melhor número de clusters...")
    melhor_k, silhuetas, inercias, k_range = encontrar_cluster(X_scaled)
    
    # K-12 é o melhor
    
    