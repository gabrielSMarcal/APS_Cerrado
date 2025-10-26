import pandas as pd
import plotly.express as px

# Substitua pelo caminho correto do arquivo CSV (certifique-se de que é o de 90 dias)
file_path = "previsao_2026_90_dias.csv"

# Carregar os dados do arquivo CSV
try:
    df = pd.read_csv(file_path)
    print(f"Dados carregados com sucesso! Total de linhas: {len(df)}")
except FileNotFoundError:
    print(f"Erro: Arquivo '{file_path}' não encontrado. Verifique o caminho.")
    exit()

# Certifique-se de que a coluna 'Data' está no formato datetime
df['Data'] = pd.to_datetime(df['Data'], errors='coerce')

# Verificar se há dados após a conversão
if df.empty:
    print("Erro: O DataFrame está vazio após carregar o CSV.")
    exit()

# Usar todos os dados (90 dias completos)
filtered_df = df

# Verificar se há dados para plotar
if filtered_df.empty:
    print("Erro: Nenhum dado encontrado.")
    exit()

print(f"Dados a serem plotados: {len(filtered_df)} linhas")
print(f"Período: {filtered_df['Data'].min()} até {filtered_df['Data'].max()}")
print(f"RiscoFogo - Mín: {filtered_df['RiscoFogo'].min()}, Máx: {filtered_df['RiscoFogo'].max()}")

# Criar o gráfico de dispersão no mapa
fig = px.scatter_mapbox(
    filtered_df,
    lat="Latitude",
    lon="Longitude",
    color="RiscoFogo",  # Usando a coluna RiscoFogo
    color_continuous_scale=['blue', 'red'],  # Degrade de azul (baixo) a vermelho (alto)
    hover_name="Data",  # Exibir a data no hover
    hover_data={
        "RiscoFogo": True,
        "DiaSemChuva": True,
        "Precipitacao": True,
        "FRP": True,
        "Latitude": False,
        "Longitude": False
    },
    mapbox_style="carto-positron",  # Estilo limpo para o Cerrado
    zoom=4.5,  # Ajuste o zoom para focar no Cerrado
    center={"lat": -15, "lon": -50},  # Centro aproximado do Cerrado
    title="Previsão de Risco de Fogo no Cerrado - 2026 (90 dias)"
)

# Personalizar o layout
fig.update_layout(
    title={
        'text': "Previsão de Risco de Fogo no Cerrado - 2026 (90 dias)",
        'x': 0.5,
        'xanchor': 'center'
    },
    coloraxis_colorbar=dict(
        title="Risco de Fogo"
    ),
    height=700
)

# Mostrar o gráfico
try:
    fig.show()
    print("Gráfico exibido com sucesso!")
except Exception as e:
    print(f"Erro ao exibir o gráfico: {e}. Tente executar em um ambiente com suporte gráfico (ex.: Jupyter Notebook ou navegador).")