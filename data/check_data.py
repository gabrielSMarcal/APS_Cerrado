from .fonte import format_csv
import pandas as pd


def check_errors():
    df = format_csv()
    drop_rows = df[df['DiaSemChuva'] == -999].index
    df = df.drop(drop_rows)
    
    drop_rows = df[df['FRP'] == -999].index
    df = df.drop(drop_rows)

    drop_rows = df[df['Precipitacao'] == -999].index
    df = df.drop(drop_rows)

    drop_rows = df[df['RiscoFogo'] == -999].index
    df = df.drop(drop_rows)

    cleaned_df = df.dropna(subset=['FRP', 'DiaSemChuva', 'Precipitacao', 'RiscoFogo'])

    return cleaned_df
