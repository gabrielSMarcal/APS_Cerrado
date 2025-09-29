import pandas as pd
import plotly.express as px

df = pd.read_csv('queimadas_cerrado_01.csv')

df = df.drop(['Satelite', 'Pais', 'Municipio', 'Bioma'], axis=1)

df['DataHora'] = pd.to_datetime(df['DataHora']).dt.date

