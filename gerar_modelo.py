from data.connection import connection
from cluster.cluster_graph import construir_grafo_hibrido
from cluster_predicao import treinar_modelo
import os

os.makedirs('./models', exist_ok=True)

print('🔄 Carregando dados históricos...')
df = connection()

if df is None or len(df) == 0:
    print('❌ Erro: Dados históricos vazios!')
    exit(1)

print(f'✅ {len(df)} registros carregados\n')

print('='*60)
print('GERANDO MODELO COM FEATURES DE GRAFO')
print('='*60)

grafo = construir_grafo_hibrido(
    df, 
    threshold_km=50.0, 
    threshold_dias=7,
    grid_size_km=25.0,
    janela_temporal_dias=7
)

treinar_modelo(
    df,
    usar_grafo=True,
    grafo=grafo,
    mostrar_acuracia=True,
    salvar_modelo=True,
    caminho_modelo='./models/modelo_completo_grafo.pkl'
)

print('\n✅ Arquivo modelo_completo_grafo.pkl gerado com sucesso!')
