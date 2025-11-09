import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# Diretório de saída
OUTPUT_DIR = Path('./assets/avaliacao_outputs')


def garantir_diretorio():
    """Cria o diretório de saída se não existir"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def gerar_feature_importance(modelo, feature_names, top_n=15, salvar=True):
    """
    Gera gráfico de importância das features do Random Forest
    
    Args:
        modelo: Modelo Random Forest treinado
        feature_names: Lista com nomes das features
        top_n: Número de features mais importantes a mostrar
        salvar: Se True, salva o gráfico
    
    Returns:
        Path do arquivo salvo ou None
    """
    garantir_diretorio()
    
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1][:top_n]
    
    # Identificar features de grafo (destacar em vermelho)
    cores = ['#E74C3C' if 'grafo_' in feature_names[i] else '#3498DB' 
             for i in indices]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(top_n)
    ax.barh(y_pos, importancias[indices], color=cores, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importância Relativa', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Features Mais Importantes - Random Forest', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', label='Features de Grafo', edgecolor='black'),
        Patch(facecolor='#3498DB', label='Features Tradicionais', edgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if salvar:
        caminho = OUTPUT_DIR / 'feature_importance_rf.png'
        plt.savefig(caminho, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Feature importance salvo: {caminho}")
        return caminho
    
    return None


def gerar_predito_vs_real(y_test, y_pred, ano=None, salvar=True):
    """
    Gera gráfico de dispersão comparando valores preditos vs reais
    
    Args:
        y_test: Valores reais (array-like)
        y_pred: Valores preditos (array-like)
        ano: Array com anos (opcional, para colorir pontos)
        salvar: Se True, salva o gráfico
    
    Returns:
        Path do arquivo salvo ou None
    """
    garantir_diretorio()
    
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # Converter para numpy arrays se necessário
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    # Scatter plot
    if ano is not None:
        scatter = ax.scatter(y_test, y_pred, c=ano, cmap='viridis', 
                            alpha=0.6, s=20, edgecolors='k', linewidth=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Ano', fontsize=11)
    else:
        ax.scatter(y_test, y_pred, alpha=0.5, s=15, color='#2C3E50', edgecolors='k', linewidth=0.2)
    
    # Linha de predição perfeita
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', lw=2.5, label='Predição Perfeita (y=x)', alpha=0.8)
    
    # Calcular R²
    r2 = r2_score(y_test, y_pred)
    
    ax.set_xlabel('Risco Real (RiscoFogo)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Risco Predito (RiscoFogo)', fontsize=12, fontweight='bold')
    ax.set_title(f'Predições vs. Valores Reais\nR² = {r2:.4f}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Tornar eixos iguais
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if salvar:
        caminho = OUTPUT_DIR / 'predicted_vs_real_scatter.png'
        plt.savefig(caminho, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Predito vs Real salvo: {caminho}")
        return caminho
    
    return None


def gerar_analise_residuos(y_test, y_pred, salvar=True):
    """
    Gera gráfico de análise de resíduos
    
    Args:
        y_test: Valores reais
        y_pred: Valores preditos
        salvar: Se True, salva o gráfico
    
    Returns:
        Path do arquivo salvo ou None
    """
    garantir_diretorio()
    
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    residuos = y_test - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Resíduos vs Predições
    ax1.scatter(y_pred, residuos, alpha=0.5, s=15, color='#E74C3C', edgecolors='k', linewidth=0.2)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax1.axhline(y=10, color='orange', linestyle=':', linewidth=1.5, label='±10 (margem)', alpha=0.7)
    ax1.axhline(y=-10, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Valores Preditos', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Resíduos (Real - Predito)', fontsize=12, fontweight='bold')
    ax1.set_title('Análise de Resíduos', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Distribuição dos Resíduos
    ax2.hist(residuos, bins=50, color='#3498DB', alpha=0.7, edgecolor='black', linewidth=0.8)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Erro Zero')
    ax2.axvline(x=residuos.mean(), color='green', linestyle='-', linewidth=2, 
                label=f'Média: {residuos.mean():.2f}')
    ax2.set_xlabel('Resíduo', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequência', fontsize=12, fontweight='bold')
    ax2.set_title('Distribuição dos Resíduos', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if salvar:
        caminho = OUTPUT_DIR / 'residuos_analise.png'
        plt.savefig(caminho, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Análise de resíduos salva: {caminho}")
        return caminho
    
    return None


def gerar_comparacao_com_sem_grafo(metricas_sem, metricas_com, salvar=True):
    """
    Gera gráfico comparando desempenho COM e SEM features de grafo
    
    Args:
        metricas_sem: Dict com métricas do modelo sem grafo
        metricas_com: Dict com métricas do modelo com grafo
        salvar: Se True, salva o gráfico
    
    Returns:
        Path do arquivo salvo ou None
    """
    garantir_diretorio()
    
    metricas_nomes = ['R²', 'MAE', 'RMSE', 'Acurácia (±10)']
    
    # Extrair valores (com defaults)
    sem_grafo = [
        metricas_sem.get('r2', 0),
        metricas_sem.get('mae', 0),
        metricas_sem.get('rmse', 0),
        metricas_sem.get('acuracia_margem_10', 0)
    ]
    com_grafo = [
        metricas_com.get('r2', 0),
        metricas_com.get('mae', 0),
        metricas_com.get('rmse', 0),
        metricas_com.get('acuracia_margem_10', 0)
    ]
    
    x = np.arange(len(metricas_nomes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width/2, sem_grafo, width, label='SEM features de grafo', 
                   color='#95A5A6', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, com_grafo, width, label='COM features de grafo', 
                   color='#27AE60', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Métricas', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor', fontsize=12, fontweight='bold')
    ax.set_title('Comparação de Desempenho: COM vs SEM Features de Grafo', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metricas_nomes)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if salvar:
        caminho = OUTPUT_DIR / 'comparacao_com_sem_grafo.png'
        plt.savefig(caminho, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Comparação COM/SEM grafo salva: {caminho}")
        return caminho
    
    return None


def gerar_erro_por_estado(df_com_erro, salvar=True):
    """
    Gera gráfico de erro médio por estado
    
    Args:
        df_com_erro: DataFrame com colunas ['Estado', 'Erro']
        salvar: Se True, salva o gráfico
    
    Returns:
        Path do arquivo salvo ou None
    """
    garantir_diretorio()
    
    # Calcular erro médio e desvio padrão por estado
    erro_por_estado = df_com_erro.groupby('Estado')['Erro'].agg(['mean', 'std']).reset_index()
    erro_por_estado = erro_por_estado.sort_values('mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colorir barras por magnitude do erro
    cores = []
    for erro in erro_por_estado['mean']:
        if abs(erro) < 5:
            cores.append('#27AE60')  # Verde - bom
        elif abs(erro) < 10:
            cores.append('#F39C12')  # Laranja - médio
        else:
            cores.append('#E74C3C')  # Vermelho - alto
    
    ax.barh(erro_por_estado['Estado'], erro_por_estado['mean'], 
            xerr=erro_por_estado['std'], color=cores, alpha=0.7, 
            edgecolor='black', linewidth=0.8,
            error_kw={'elinewidth': 1.5, 'alpha': 0.5, 'capsize': 3})
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Erro Médio (Real - Predito)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Estado', fontsize=12, fontweight='bold')
    ax.set_title('Erro de Predição por Estado\n(Negativo = Superestimação | Positivo = Subestimação)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27AE60', label='Erro < 5 (Excelente)', edgecolor='black'),
        Patch(facecolor='#F39C12', label='Erro 5-10 (Bom)', edgecolor='black'),
        Patch(facecolor='#E74C3C', label='Erro > 10 (Atenção)', edgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if salvar:
        caminho = OUTPUT_DIR / 'erro_por_estado.png'
        plt.savefig(caminho, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Erro por estado salvo: {caminho}")
        return caminho
    
    return None


def gerar_todas_visualizacoes(modelo, X, y_test, y_pred, df_test=None):
    """
    Função principal que gera todas as visualizações de uma vez
    
    Args:
        modelo: Modelo Random Forest treinado
        X: DataFrame com features (para nomes)
        y_test: Valores reais
        y_pred: Valores preditos
        df_test: DataFrame de teste com coluna 'Estado' (opcional)
    
    Returns:
        Dict com caminhos de todos os arquivos gerados
    """
    print("\n" + "="*60)
    print("GERANDO VISUALIZAÇÕES DO MODELO")
    print("="*60 + "\n")
    
    caminhos = {}
    
    # 1. Feature Importance
    caminhos['feature_importance'] = gerar_feature_importance(
        modelo, list(X.columns), top_n=15
    )
    
    # 2. Predito vs Real
    ano = df_test['Ano'].values if df_test is not None and 'Ano' in df_test.columns else None
    caminhos['predito_vs_real'] = gerar_predito_vs_real(y_test, y_pred, ano=ano)
    
    # 3. Análise de Resíduos
    caminhos['residuos'] = gerar_analise_residuos(y_test, y_pred)
    
    # 4. Erro por Estado (se disponível)
    if df_test is not None and 'Estado' in df_test.columns:
        df_erro = pd.DataFrame({
            'Estado': df_test['Estado'].values,
            'Erro': y_test - y_pred
        })
        caminhos['erro_estado'] = gerar_erro_por_estado(df_erro)
    
    print("\n" + "="*60)
    print("✅ TODAS AS VISUALIZAÇÕES FORAM GERADAS COM SUCESSO!")
    print("="*60 + "\n")
    
    return caminhos


# ============================================================================
# EXEMPLO DE INTEGRAÇÃO
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  MÓDULO DE VISUALIZAÇÕES - APS_CERRADO                       ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Para integrar ao seu código existente em 'prev/cluster_predicao.py',
    adicione após treinar o modelo:
    
    ───────────────────────────────────────────────────────────────
    from render.visualizacoes_modelo import gerar_todas_visualizacoes
    
    # Após fazer as predições
    y_pred = modelo.predict(X_test_scaled)
    
    # Gerar todas as visualizações
    gerar_todas_visualizacoes(
        modelo=modelo,
        X=X_train,  # ou X (antes do split)
        y_test=y_test,
        y_pred=y_pred,
        df_test=df_test  # DataFrame original do conjunto de teste
    )
    ───────────────────────────────────────────────────────────────
    """)
