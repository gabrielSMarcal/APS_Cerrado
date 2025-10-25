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

df = c.connection()

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
    
    # Verificar se há valores não numéricos
    print(f"Colunas em X: {X.columns.tolist()}")
    print(f"Tipos de dados em X:\n{X.dtypes}")
    
    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, X_scaled, y, scaler

def encontrar_cluster(X_scaled, k_range=range(2, 20)):
    
    inercias = []
    silhuetas = []
    
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
    plt.show()
    
    melhor_k = k_range[np.argmax(silhuetas)]
    print(f'Melhor número de clusters (k) baseado na silhueta: {melhor_k}')
    print(f'Silhouette Score em k=12 {silhuetas[10]:.4f}')
    
    return melhor_k, silhuetas

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
    
def visualizar_clusters(X_scaled, clusters, y):
    """
    Visualiza os clusters criados
    """
    # Se tiver muitas dimensões, usar PCA para visualização
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Clusters
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Primeira Componente Principal')
    plt.ylabel('Segunda Componente Principal')
    plt.title('Visualização dos Clusters (PCA)')
    
    # Plot 2: Risco de Fogo
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='YlOrRd', alpha=0.6)
    plt.colorbar(scatter, label='RiscoFogo')
    plt.xlabel('Primeira Componente Principal')
    plt.ylabel('Segunda Componente Principal')
    plt.title('Risco de Fogo no Espaço PCA')
    
    plt.tight_layout()
    plt.show()

def analisar_clusters(df_original, clusters, y):
    """
    Analisa as características de cada cluster
    """
    df_analise = df_original.copy()
    df_analise['Cluster'] = clusters
    df_analise['RiscoFogo'] = y
    
    # Estatísticas por cluster
    print("\n=== Análise por Cluster ===")
    for cluster_id in sorted(df_analise['Cluster'].unique()):
        print(f"\n--- Cluster {cluster_id} ---")
        cluster_data = df_analise[df_analise['Cluster'] == cluster_id]
        print(f"Tamanho: {len(cluster_data)} registros")
        print(f"RiscoFogo médio: {cluster_data['RiscoFogo'].mean():.2f}")
        print(f"RiscoFogo mediano: {cluster_data['RiscoFogo'].median():.2f}")
        print(f"RiscoFogo min-max: {cluster_data['RiscoFogo'].min():.2f} - {cluster_data['RiscoFogo'].max():.2f}")
        
        # Meses mais comuns neste cluster
        colunas_mes = [col for col in df_analise.columns if col.startswith('Mes_')]
        if colunas_mes:
            meses_ativos = cluster_data[colunas_mes].sum().sort_values(ascending=False).head(3)
            print(f"Meses mais frequentes: {meses_ativos.index.tolist()}")
    
    return df_analise

# Executar o pipeline de clustering
if __name__ == "__main__":
    print("Preparando dados...")
    X, X_scaled, y, scaler = preparar_dados(df)
    
    print("\nEncontrando melhor número de clusters...")
    melhor_k = encontrar_cluster(X_scaled)
    
    print(f"\nCriando clusters com k={melhor_k}...")
    kmeans_model, clusters = criar_clusters(X_scaled, melhor_k)
    
    print("\nVisualizando clusters...")
    visualizar_clusters(X_scaled, clusters, y)
    
    print("\nAnalisando características dos clusters...")
    df_com_clusters = analisar_clusters(X, clusters, y)
    
    # Salvar modelo e scaler para uso posterior em previsão
    import pickle
    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans_model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\nModelo e scaler salvos com sucesso!")