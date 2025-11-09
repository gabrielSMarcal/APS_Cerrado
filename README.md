# 🔥 Dashboard de Análise de Risco de Fogo - Cerrado Brasileiro

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Dash](https://img.shields.io/badge/dash-3.2.0-brightgreen)
![License](https://img.shields.io/badge/license-MIT-orange)

## 📋 Visão Geral

Dashboard interativo desenvolvido em **Dash** para análise preditiva de risco de incêndios no Cerrado Brasileiro. O sistema utiliza **Machine Learning** (K-means clustering e Random Forest) para prever padrões de risco baseado em dados históricos do INPE e variáveis climáticas.

### ✨ Funcionalidades Principais

- 📊 **Visualização Histórica**: Gráficos interativos de focos de incêndio por ano (2014-2025).
- 🗺️ **Mapas de Calor**: Heatmaps geográficos com intensidade de risco por região.
- 🤖 **Modelo Preditivo**: Random Forest treinado com clustering e features de grafo para previsão de risco.
- 📈 **Métricas de Desempenho**: Avaliação completa do modelo (MAE, RMSE, R², Acurácia).
- 🔮 **Previsão 2026**: Projeção de risco para o ano seguinte com filtros por estado.
- 🕸️ **Análise de Grafos**: Conexões espaço-temporais entre focos de incêndio para enriquecer o modelo.

---

## 🏗️ Estrutura do Projeto

A estrutura foi organizada de forma modular para separar as responsabilidades da aplicação, facilitando a manutenção e escalabilidade.

```
APS_Cerrado/
├── 📄 app.py              # Inicializador da aplicação Dash
├── 📄 main.py             # Ponto de entrada para executar o servidor
├── 📄 gerar_modelo.py     # Script para treinar o modelo de ML
├── 📄 gerar_previsao.py   # Script para gerar o arquivo de previsão para 2026
│
├── 📁 pages/              # Módulos de cada página do dashboard (home, graficos, etc.)
│
├── 📁 data/                # Módulos de conexão e manipulação de dados
│   ├── base_db/          # CSVs com dados históricos brutos por ano
│   └── treated_db/       # CSVs com dados consolidados e tratados
│
├── 📁 cluster/             # Lógica de pré-processamento e clustering
│
├── 📁 prev/                # Lógica de previsão (Machine Learning)
│
├── 📁 models/              # Classes e Tipos Abstratos de Dados (TADs)
│   └── TAD/              # Implementação de Grafos
│
├── 📁 render/              # Scripts para gerar saídas visuais (gráficos, métricas)
│
├── 📁 source/              # Artefatos gerados (modelo .pkl e previsão .csv)
│
├── 📁 assets/              # Arquivos estáticos (CSS, imagens, etc.)
│   └── avaliacao_outputs/ # Gráficos e métricas de avaliação do modelo
│
├── 📄 requirements.txt    # Dependências do projeto
└── 📄 README.md           # Esta documentação
```

---

## 🚀 Instalação e Execução

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### 1. Clone o Repositório

```bash
git clone https://github.com/gabrielSMarcal/APS_Cerrado.git
cd APS_Cerrado
```

### 2. Instale as Dependências

```bash
pip install -r requirements.txt
```

### 3. Prepare os Dados

Certifique-se de que os CSVs históricos (2014-2025) estão na pasta `data/base_db/`. O sistema consolidará os dados automaticamente quando necessário através do módulo `data/connection.py`.

### 4. (Opcional) Treine o Modelo de Machine Learning

Para gerar um novo arquivo de modelo (`.pkl`) a partir dos dados históricos:

```bash
python gerar_modelo.py
```

Este script irá executar o pipeline de treinamento e salvar o artefato em `source/modelo_random_forest.pkl`.

### 5. (Opcional) Gere as Previsões para 2026

Para gerar o arquivo `previsao_2026.csv` usando o modelo treinado:

```bash
python gerar_previsao.py
```

Este script utiliza o modelo salvo para criar um cenário futuro e prever os riscos, salvando o resultado em `source/previsao_2026.csv`.

### 6. Inicie o Dashboard

```bash
python main.py
```

Acesse no navegador: **http://127.0.0.1:8050**

---

## 📊 Páginas do Dashboard

- **🏠 Página Inicial (`/`)**: Introdução ao projeto e navegação para as demais seções.
- **📈 Gráficos por Ano (`/graficos`)**: Mapas de dispersão interativos para os dados históricos (2014-2025), com filtros por ano e visualização da média de risco por estado.
- **🧮 Modelo de Previsão (`/dados`)**: Explicação da metodologia de Machine Learning e apresentação das métricas de desempenho do modelo (MAE, RMSE, R², Acurácia) de forma interativa.
- **🗺️ Previsão 2026 (`/mapa`)**: Mapa interativo com as previsões de risco de fogo para 2026, com filtros por estado e um painel de estatísticas detalhadas.

---

## 🤖 Modelo de Machine Learning

### Arquitetura

1.  **Pré-processamento (`cluster/preparacao_dados.py`)**: Criação de variáveis temporais, encoding de variáveis categóricas e normalização de features.
2.  **Análise de Grafos (`models/TAD/ClusterGraph.py`)**: Construção de um grafo espaço-temporal para extrair features de conectividade (grau, centralidade), enriquecendo os dados.
3.  **K-means Clustering (`cluster/cluster.py`)**: Usado para análise exploratória e para encontrar o número ideal de clusters, validado pelo método da silhueta.
4.  **Random Forest Regressor (`prev/cluster_predicao.py`)**: Modelo principal que prevê o `RiscoFogo` (0-100), treinado com os dados enriquecidos pelo grafo.

### Variáveis Utilizadas

-   **Climáticas**: `DiaSemChuva`, `Precipitacao`, `FRP` (Fire Radiative Power).
-   **Espaciais**: `Latitude`, `Longitude`, `Estado`, `Municipio`.
-   **Temporais**: `Ano`, `DiaAno`, e dummies para os meses.
-   **Grafo** (opcional): `grau`, `centralidade`, `clustering_coef`.

---

## 📦 Principais Dependências

-   `dash`: Framework web
-   `dash-bootstrap-components`: Componentes Bootstrap
-   `plotly`: Gráficos interativos
-   `pandas`: Manipulação de dados
-   `scikit-learn`: Machine Learning
-   `folium`: Mapas interativos (usado na geração de heatmaps de avaliação)

Ver `requirements.txt` para a lista completa.

---

## 🙏 Agradecimentos

-   **INPE**: Fornecimento dos dados de focos de incêndio.
-   Comunidades **Plotly/Dash** e **scikit-learn**.

---

**⚠️ Disclaimer**: Este é um projeto com fins educacionais e acadêmicos. As previsões geradas são baseadas em padrões históricos e devem ser validadas por especialistas antes de qualquer uso operacional.
