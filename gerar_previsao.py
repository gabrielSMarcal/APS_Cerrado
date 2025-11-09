import os
from prev.cluster_gerar_csv import main

# Criar diretório de saída se não existir
os.makedirs('./source/test', exist_ok=True)

print('='*80)
print('GERADOR DE PREVISÕES PARA 2026 - RISCO DE INCÊNDIO NO CERRADO')
print('Sistema com TAD (Grafo) para análise espacial e temporal')
print('='*80)

main(usar_grafo=True)

print('\n✅ Processo concluído com sucesso!')
print('📁 Arquivo gerado: ./source/test/previsao_2026.csv')