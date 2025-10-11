import plotly.express as px
from data import fonte

csvpath = "./data/queimadas_cerrado_01.csv"

df = fonte.format_csv(csvpath=csvpath) 

fig = px.scatter_mapbox(
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
    mapbox_style="carto-positron",
    zoom=4
)
fig.show()
