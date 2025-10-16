from .connection import connection
import pandas as pd


def check_errors():
    df, list_of_dfs = connection()
    drop_rows = df[df['DiaSemChuva'] == -999].index
    df = df.drop(drop_rows)
    
    drop_rows = df[df['FRP'] == -999].index
    df = df.drop(drop_rows)

    drop_rows = df[df['Precipitacao'] == -999].index
    df = df.drop(drop_rows)

    drop_rows = df[df['RiscoFogo'] == -999].index
    df = df.drop(drop_rows)

    cleaned_df = df.dropna(subset=['FRP', 'DiaSemChuva', 'Precipitacao', 'RiscoFogo'])


    clean_list_dfs = []
    for dfs in list_of_dfs:
        clean_df = dfs
        drop_rows = clean_df[clean_df['DiaSemChuva'] == -999].index
        clean_df = clean_df.drop(drop_rows)
        
        drop_rows = clean_df[clean_df['FRP'] == -999].index
        clean_df = clean_df.drop(drop_rows)

        drop_rows = clean_df[clean_df['Precipitacao'] == -999].index
        clean_df = clean_df.drop(drop_rows)

        drop_rows = clean_df[clean_df['RiscoFogo'] == -999].index
        clean_df = clean_df.drop(drop_rows)

        clean_list_dfs.append(clean_df.dropna(subset=['FRP', 'DiaSemChuva', 'Precipitacao', 'RiscoFogo']))

    cleaned_df.to_csv('./data/treated_db/db_cerrado_cleaned.csv', index=False)
    return cleaned_df, clean_list_dfs
