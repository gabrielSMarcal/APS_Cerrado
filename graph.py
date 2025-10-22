import plotly.express as px
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from data.connection import connection, get_df_list

df = connection()
df_list = get_df_list()

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.DatePickerRange(
        id='date-range-slider',
        min_date_allowed=df['Data'].min(),
        max_date_allowed=df['Data'].max(),
        start_date=df['Data'].min(),
        end_date=df['Data'].max(),
    ),
    dcc.Graph(id='forest_burn_map')
])

@app.callback(
    Output('forest_burn_map', 'figure'),
    Input('date-range-slider', 'start_date'),
    Input('date-range-slider', 'end_date')
)
def update_map(start_date, end_date):
    if not start_date or not end_date:
        return dash.no_update

    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()

    filtered_df = df[(df['Data'] >= start) & (df['Data'] <= end)]

    fig = px.scatter_map(
        filtered_df,
        lat="Latitude",      
        lon="Longitude",    
        color="RiscoFogo",
        color_continuous_scale=px.colors.sequential.Turbo,
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
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)

for df in df_list:
    fig = px.scatter_map(
        df,
        lat="Latitude",      
        lon="Longitude",    
        color="RiscoFogo",
        color_continuous_scale=px.colors.sequential.Turbo,
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



# fig = px.scatter(
#     df,
#     color='RiscoFogo',
#     y='DataHora',
#     x='DiaSemChuva'
# )
#
#
# fig = px.scatter(
#     df,
#     color='RiscoFogo',
#     y='DataHora',
#     x='FRP'
# )
#
#
# fig = px.scatter(
#     df,
#     color='RiscoFogo',
#     y='DataHora',
#     x='Precipitacao'
# )
