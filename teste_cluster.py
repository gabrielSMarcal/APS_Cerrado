import pandas as pd
import plotly.express as px

# Substitua pelo caminho correto do arquivo CSV (certifique-se de que é o de 365 dias ou 90 dias conforme deseja)
file_path = "previsao_2026.csv"

# Carregar como texto para tratar possíveis linhas de cabeçalho repetidas
try:
    df = pd.read_csv(file_path, dtype=str, skip_blank_lines=True)
    print(f"Arquivo lido como texto. Linhas iniciais: {len(df)}")
except FileNotFoundError:
    print(f"Erro: Arquivo '{file_path}' não encontrado. Verifique o caminho.")
    exit()

# Remover linhas que contenham um cabeçalho repetido no meio do CSV (ex.: "Data,Latitude,...")
if 'Data' in df.columns:
    # se por algum motivo a coluna "Data" contiver a string "Data" em registros, remover essas linhas
    df = df[df['Data'].str.strip().ne('Data')]

# Converter colunas para tipos apropriados
cols_num = ["Latitude", "Longitude", "RiscoFogo", "DiaSemChuva", "Precipitacao", "FRP"]
for c in cols_num:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Converter Data para datetime
if 'Data' in df.columns:
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')

# Dropar registros sem coordenadas válidas ou sem data
before = len(df)
df = df.dropna(subset=['Latitude', 'Longitude', 'Data'])
after = len(df)
print(f"Linhas com lat/lon/data válidos: {after} (removidas {before - after})")

if df.empty:
    print("Erro: sem dados válidos após limpeza.")
    exit()

# Ordenar por data (ajuda a verificar se todos os dias foram mantidos)
df = df.sort_values('Data').reset_index(drop=True)

# Mostrar resumo de datas para diagnóstico
unique_dates = df['Data'].dt.date.unique()
print(f"Total de linhas a serem plotadas: {len(df)}")
print(f"Dias distintos presentes no arquivo: {len(unique_dates)}. Primeiros/últimos: {unique_dates[:3]} ... {unique_dates[-3:]}")

# Criar o gráfico de dispersão no mapa (mesma configuração que já tinha)
fig = px.scatter_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    color="RiscoFogo",
    color_continuous_scale=px.colors.sequential.Turbo,
    hover_name="Data",
    hover_data={
        "Estado": True,
        "Municipio": True,
        "RiscoFogo": True,
        "DiaSemChuva": True,
        "Precipitacao": True,
        "FRP": True,
        "Latitude": False,
        "Longitude": False
    },
    mapbox_style="carto-positron",
    zoom=4.5,
    center={"lat": -15, "lon": -50},
    title="Previsão de Risco de Fogo no Cerrado - 2026"
)

fig.update_layout(
    title={
        'text': "Previsão de Risco de Fogo no Cerrado - 2026",
        'x': 0.5,
        'xanchor': 'center'
    },
    coloraxis_colorbar=dict(title="Risco de Fogo"),
    height=700
)

try:
    fig.show()
    print("Gráfico exibido com sucesso!")
except Exception as e:
    print(f"Erro ao exibir o gráfico: {e}. Tente executar em um ambiente com suporte gráfico (ex.: Jupyter Notebook ou navegador).")