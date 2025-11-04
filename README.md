# 🔥 Dashboard de Análise de Risco de Fogo - Cerrado Brasileiro

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Dash](https://img.shields.io/badge/dash-3.2.0-brightgreen)
![License](https://img.shields.io/badge/license-MIT-orange)

## 📋 Visão Geral

Dashboard interativo desenvolvido em **Dash** para análise preditiva de risco de incêndios no Cerrado Brasileiro. O sistema utiliza **Machine Learning** (K-means clustering e Random Forest) para prever padrões de risco baseado em dados históricos do INPE e variáveis climáticas.

### ✨ Funcionalidades Principais

- 📊 **Visualização Histórica**: Gráficos interativos de focos de incêndio por ano (2014-2025)
- 🗺️ **Mapas de Calor**: Heatmaps geográficos com intensidade de risco por região
- 🤖 **Modelo Preditivo**: Random Forest treinado com clustering para previsão de risco
- 📈 **Métricas de Desempenho**: Avaliação completa do modelo (MAE, RMSE, R², Acurácia)
- 🔮 **Previsão 2026**: Projeção de risco para o ano seguinte com filtros por estado
- 📊 **Análise de Grafos**: Conexões espaço-temporais entre focos de incêndio

---

## 🏗️ Estrutura do Projeto

```
Cerrado/
├── 📁 app.py                      # Inicialização do aplicativo Dash
├── 📁 main.py                     # Roteamento e layout principal
├── 📁 cluster_gerar_previsao.py   # Script de geração de previsões 2026
├── 📁 grafico_acuracia.py         # Geração de métricas e gráficos de avaliação
├── 📁 lista_grafos.py             # Geração de gráficos por ano
│
├── 📂 pages/                      # Páginas do dashboard
│   ├── home.py                    # Página inicial
│   ├── graficos.py                # Visualização histórica por ano
│   ├── dados.py                   # Métricas e desempenho do modelo
│   └── mapa.py                    # Mapa interativo de previsão 2026
│
├── 📂 data/                       # Dados e conexões
│   ├── connection.py              # Gerenciamento de conexões com CSVs
│   ├── check_data.py              # Validação de dados
│   ├── fonte.py                   # Formatação de dados
│   ├── base_db/                   # CSVs históricos por ano (2014-2025)
│   └── treated_db/                # Dados tratados e previsões
│       ├── db_cerrado.csv         # Base consolidada
│       └── previsao_2026.csv      # Previsão gerada
│
├── 📂 cluster/                    # Machine Learning
│   ├── cluster.py                 # K-means clustering
│   ├── cluster_predicao.py        # Random Forest para predição
│   ├── cluster_graph.py           # Análise de grafos espaço-temporais
│   ├── cluster_utils.py           # Funções auxiliares
│   └── predicao.py                # Funções de predição
│
├── 📂 models/                     # Classes e TADs
│   ├── Graph.py                   # TAD Grafo base
│   ├── ClusterGraph.py            # Grafo para análise de clusters
│   └── MapaInterativo.py          # Classe de mapa interativo encapsulado
│
├── 📂 avaliacao_outputs/          # Resultados de avaliação
│   ├── metrics_continuas_por_ano.csv
│   ├── metrics_margem_por_ano.csv
│   └── heatmap_*.html             # Heatmaps por ano
│
├── 📂 assets/                     # Arquivos estáticos
│   └── main.css                   # Estilos personalizados
│
├── 📄 requirements.txt            # Dependências do projeto
└── 📄 README.md                   # Documentação
```

---

## 🚀 Instalação e Execução

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### 1. Clone o Repositório

```bash
git clone <url-do-repositorio>
cd Cerrado
```

### 2. Instale as Dependências

```bash
pip install -r requirements.txt
```

### 3. Prepare os Dados

Certifique-se de que os CSVs históricos estão em [`data/base_db/`](data/base_db/). O sistema consolidará automaticamente em [`data/treated_db/db_cerrado.csv`](data/treated_db/db_cerrado.csv).

### 4. (Opcional) Gere Previsões para 2026

```bash
python cluster_gerar_previsao.py
```

Este script:
- Analisa padrões históricos (2014-2025)
- Gera dados sintéticos inteligentes para 2026 (45k-60k registros)
- Aplica o modelo treinado para prever risco
- Salva resultado em [`data/treated_db/previsao_2026.csv`](data/treated_db/previsao_2026.csv)

### 5. (Opcional) Gere Métricas de Avaliação

```bash
python grafico_acuracia.py
```

Este script gera:
- Métricas por ano (MAE, RMSE, R², Acurácia)
- Gráficos de desempenho
- Heatmaps HTML por ano em [`avaliacao_outputs/`](avaliacao_outputs/)

### 6. Inicie o Dashboard

```bash
python main.py
```

Acesse no navegador: **http://127.0.0.1:8050**

---

## 📊 Páginas do Dashboard

### 🏠 Página Inicial ([`/`](pages/home.py))
- Introdução ao projeto
- Navegação para outras seções

### 📈 Gráficos por Ano ([`/graficos`](pages/graficos.py))
- Mapas de dispersão interativos (2014-2025)
- Visualização de focos de incêndio por estado
- Filtros por ano

### 🧮 Modelo de Previsão ([`/dados`](pages/dados.py))
- Explicação do modelo (K-means + Random Forest)
- Métricas de desempenho por ano:
  - **MAE** (Mean Absolute Error)
  - **RMSE** (Root Mean Square Error)
  - **R²** (Coeficiente de Determinação)
  - **Acurácia** (±10 pontos de margem)
- Gráficos de evolução temporal
- Heatmaps de risco por ano

### 🗺️ Previsão 2026 ([`/mapa`](pages/mapa.py))
- Mapa interativo de previsão
- Filtros por estado do Cerrado
- Estatísticas em tempo real:
  - Total de registros
  - Municípios afetados
  - Distribuição por nível de risco:
    - 🔴 Baixo (0-20): possível incêndio criminoso
    - 🟡 Médio (21-70): condições favoráveis
    - 🟢 Alto (71-100): incêndio natural provável

---

## 🤖 Modelo de Machine Learning

### Arquitetura

1. **Pré-processamento**
   - Criação de variáveis temporais (mês, dia do ano)
   - Encoding de variáveis categóricas (Estado, Município)
   - Normalização de features numéricas

2. **K-means Clustering** ([`cluster/cluster.py`](cluster/cluster.py))
   - Agrupa anos em 12 clusters (sazonalidade mensal)
   - Análise de silhueta e método do cotovelo
   - Opção de usar features de grafo espaço-temporal

3. **Random Forest Regressor** ([`cluster/cluster_predicao.py`](cluster/cluster_predicao.py))
   - Predição de `RiscoFogo` (0-100)
   - Treinamento com validação temporal
   - Feature importance analysis

### Variáveis Utilizadas

**Features climáticas:**
- `DiaSemChuva`: dias consecutivos sem precipitação
- `Precipitacao`: precipitação acumulada (mm)
- `FRP`: Fire Radiative Power

**Features espaciais:**
- `Latitude`, `Longitude`: coordenadas geográficas
- `Estado`, `Municipio`: localização administrativa

**Features temporais:**
- `Mes_1` a `Mes_12`: variáveis dummy para sazonalidade
- `Ano`, `DiaAno`: padrões temporais

**Features de grafo** (opcional):
- `grau`: número de conexões espaciais/temporais
- `centralidade`: importância do ponto no grafo
- `clustering_coef`: coeficiente de agrupamento local

---

## 📦 Principais Dependências

```
dash==3.2.0                    # Framework web
dash-bootstrap-components      # Componentes Bootstrap
plotly==6.3.0                  # Gráficos interativos
pandas==2.3.2                  # Manipulação de dados
numpy==2.3.3                   # Operações numéricas
scikit-learn==1.7.2            # Machine Learning
folium==0.20.0                 # Mapas interativos
matplotlib==3.10.7             # Visualizações estáticas
```

Ver [`requirements.txt`](requirements.txt) para lista completa.

---

## 🔧 Scripts Utilitários

### [`cluster_gerar_previsao.py`](cluster_gerar_previsao.py)
Gera previsões inteligentes para 2026:
```bash
python cluster_gerar_previsao.py
```

### [`grafico_acuracia.py`](grafico_acuracia.py)
Avalia desempenho do modelo:
```bash
python grafico_acuracia.py
```

### [`lista_grafos.py`](lista_grafos.py)
Gera lista de gráficos por ano (usado internamente).

---

## 📈 Estrutura de Dados

### CSV Principal: [`db_cerrado.csv`](data/treated_db/db_cerrado.csv)

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `Data` | datetime | Data do registro |
| `Latitude` | float | Latitude (graus) |
| `Longitude` | float | Longitude (graus) |
| `Estado` | string | Estado do Cerrado |
| `Municipio` | string | Município |
| `DiaSemChuva` | int | Dias sem precipitação |
| `Precipitacao` | float | Precipitação (mm) |
| `FRP` | float | Fire Radiative Power |
| `RiscoFogo` | int | Risco de fogo (0-100) |

---

## 🧪 Testes e Validação

O modelo é validado usando:
- **Validação temporal**: treino em anos anteriores, teste em anos seguintes
- **Margem de erro**: acurácia com ±10 pontos de tolerância
- **Métricas contínuas**: MAE, RMSE, R²

Resultados salvos em [`avaliacao_outputs/`](avaliacao_outputs/).

---

## 🎨 Personalização

### Estilos CSS

Edite [`assets/main.css`](assets/main.css) para customizar:
- Cores do tema
- Fontes e tipografia
- Layout responsivo

### Adicionar Novas Páginas

1. Crie arquivo em [`pages/`](pages/)
2. Importe em [`pages/__init__.py`](pages/__init__.py)
3. Adicione rota em [`main.py`](main.py)

---

## 🙏 Agradecimentos

- **INPE** - Dados de focos de incêndio
- Comunidade **Plotly/Dash** - Framework web
- Comunidade **scikit-learn** - Ferramentas de ML

---

**⚠️ Disclaimer**: Este é um projeto educacional/acadêmico. As previsões devem ser validadas com especialistas antes de uso operacional.