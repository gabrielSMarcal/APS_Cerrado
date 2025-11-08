import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Optional, Tuple

from models.TAD.ClusterGraph import ClusterGraph
from cluster.preparacao_dados import preparar_para_clustering


def encontrar_cluster(X_scaled, k_range=range(2, 20), salvar_grafico: bool = True):
    '''
    Na escala de 2 a 20 clusters, encontrar o melhor número de clusters
    usando os métodos do cotovelo e da silhueta.
    '''
    
    inercias = []
    silhuetas = []
    
    print(f'Linhas em X_scaled: {X_scaled.shape[0]}')
    
    for k in k_range:
        print(f'Calculando para k={k}...')
        kmeans = KMeans(n_clusters=k, random_state=657, n_init='auto')
        kmeans.fit(X_scaled)
        inercias.append(kmeans.inertia_)
        silhuetas.append(silhouette_score(X_scaled, kmeans.labels_))
        print(f' - Inércia: {kmeans.inertia_:.2f}, Silhouette Score: {silhuetas[-1]:.4f}')
        
    if salvar_grafico:
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
        
        # Salvar gráfico
        plt.savefig('graficos_clustering.png', dpi=300, bbox_inches='tight')
        print("Gráfico salvo em 'graficos_clustering.png'")
        
        plt.close()
    
    melhor_k = k_range[np.argmax(silhuetas)]
    print(f"Melhor número de clusters (k) baseado na silhueta: {melhor_k}")
    if len(silhuetas) > 10:
        print(f"Silhouette Score em k=12: {silhuetas[10]:.4f}")
    
    return melhor_k, silhuetas, inercias, k_range


def aplicar_clustering(df, n_clusters: int = 12, usar_grafo: bool = False, grafo: Optional[ClusterGraph] = None) -> Tuple[pd.DataFrame, KMeans, StandardScaler, list]:
    '''
    Aplica clustering usando módulo de preparação centralizado.
    '''

    print(f"\n{'='*60}")
    print(f"APLICANDO CLUSTERING (k={n_clusters})")
    if usar_grafo:
        print(f"Modo: COM features de grafo")
    else:
        print(f"Modo: SEM features de grafo (original)")
    print(f"{'='*60}\n")
    
    # Usar função centralizada
    X, X_scaled, y, label_encoders, feature_names = preparar_para_clustering(
        df, usar_grafo=usar_grafo, grafo=grafo
    )
    
    # Aplicar KMeans
    print(f"\nTreinando KMeans com {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=657, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calcular score de silhueta
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    
    # Adicionar cluster_id ao DataFrame
    df_resultado = df.copy()
    df_resultado['cluster_id'] = cluster_labels
    
    # Estatísticas dos clusters
    print(f"\nDistribuição dos clusters:")
    print(df_resultado['cluster_id'].value_counts().sort_index())
    
    # Criar scaler para retornar (compatibilidade)
    scaler = StandardScaler()
    scaler.fit(X)
    
    return df_resultado, kmeans, scaler, feature_names


def comparar_clustering_com_sem_grafo(df, grafo: ClusterGraph, n_clusters: int = 12) -> dict:
    '''
    Compara performance do clustering com e sem features de grafo.
    '''
    
    print(f"\n{'='*60}")
    print(f"COMPARAÇÃO: CLUSTERING COM vs SEM GRAFO")
    print(f"{'='*60}\n")

    # Clustering sem grafo (original)
    df_sem_grafo, kmeans_sem, scaler_sem, features_sem = aplicar_clustering(
        df, n_clusters, usar_grafo=False, grafo=None
    )
    
    # Clustering com grafo
    df_com_grafo, kmeans_com, scaler_com, features_com = aplicar_clustering(
        df, n_clusters, usar_grafo=True, grafo=grafo
    )
    
    # Preparar dados para cálculo de silhueta usando função centralizada
    X_sem, X_scaled_sem, _, _, _ = preparar_para_clustering(
        df, usar_grafo=False, grafo=None
    )
    
    X_com, X_scaled_com, _, _, _ = preparar_para_clustering(
        df, usar_grafo=True, grafo=grafo
    )
    
    # Calcular scores
    silhouette_sem = silhouette_score(X_scaled_sem, df_sem_grafo['cluster_id'])
    silhouette_com = silhouette_score(X_scaled_com, df_com_grafo['cluster_id'])
    
    resultados = {
        'silhouette_sem_grafo': silhouette_sem,
        'silhouette_com_grafo': silhouette_com,
        'melhoria': silhouette_com - silhouette_sem,
        'melhoria_percentual': ((silhouette_com - silhouette_sem) / silhouette_sem) * 100,
        'num_features_sem': len(features_sem),
        'num_features_com': len(features_com),
        'features_adicionadas': len(features_com) - len(features_sem)
    }

    print(f"\n{'='*60}")
    print(f"RESULTADOS DA COMPARAÇÃO")
    print(f"{'='*60}")
    print(f"SEM Grafo:")
    print(f"  - Silhouette Score: {silhouette_sem:.4f}")
    print(f"  - Número de features: {len(features_sem)}")
    print(f"\nCOM Grafo:")
    print(f"  - Silhouette Score: {silhouette_com:.4f}")
    print(f"  - Número de features: {len(features_com)}")
    print(f"  - Features adicionadas: {len(features_com) - len(features_sem)}")
    print(f"\nMelhoria:")
    print(f"  - Absoluta: {resultados['melhoria']:+.4f}")
    print(f"  - Percentual: {resultados['melhoria_percentual']:+.2f}%")
    print(f"{'='*60}\n")

    return resultados


def visualizar_clusters_pca(
    df, 
    usar_grafo: bool = False, 
    grafo: Optional[ClusterGraph] = None, 
    n_clusters: int = 12, 
    salvar: bool = True
) -> None:
    '''
    Visualiza clusters em 2D usando PCA.
    '''
    
    # Aplicar clustering
    df_clustered, kmeans, scaler, feature_names = aplicar_clustering(
        df, n_clusters, usar_grafo, grafo
    )
    
    # Preparar dados usando função centralizada
    X, X_scaled, y, _, _ = preparar_para_clustering(
        df, usar_grafo=usar_grafo, grafo=grafo
    )
    
    # Aplicar PCA para reduzir a 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Criar gráfico
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=df_clustered['cluster_id'],
        cmap='tab20',
        alpha=0.6,
        edgecolors='k',
        linewidth=0.5
    )
    
    plt.colorbar(scatter, label='Cluster ID')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variância)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variância)')
    
    titulo = f'Visualização dos Clusters (k={n_clusters})'
    if usar_grafo:
        titulo += ' - COM features de grafo'
    else:
        titulo += ' - SEM features de grafo'
    
    plt.title(titulo)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if salvar:
        nome_arquivo = f'clusters_pca_{"com" if usar_grafo else "sem"}_grafo.png'
        plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico salvo em '{nome_arquivo}'")
    
    plt.close()