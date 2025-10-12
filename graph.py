import plotly.express as px
from data import check_data


df = check_data.check_errors()

print("gerando mapa")
fig = px.scatter_map(
    df,
    lat="Latitude",      
    lon="Longitude",    
    color="RiscoFogo",
    hover_name="Estado",
    hover_data={
        "Municipio": True,
        "DataHora": True,
        "DiaSemChuva": True,
        "Precipitacao": True,
        "FRP": True,
        "Latitude": False,
        "Longitude": False
    },
    map_style="carto-positron",
    zoom=4
)
fig.show()

fig = px.scatter(
    df,
    color='RiscoFogo',
    y='DataHora',
    x='DiaSemChuva'
)

fig.show()

fig = px.scatter(
    df,
    color='RiscoFogo',
    y='DataHora',
    x='FRP'
)

fig.show()

fig = px.scatter(
    df,
    color='RiscoFogo',
    y='DataHora',
    x='Precipitacao'
)

fig.show()

