import pandas as pd

def check_errors(df):
    drop_rows = df[df['DiaSemChuva'] == -999].index
    df = df.drop(drop_rows)
    
    drop_rows = df[df['FRP'] == -999].index
    df = df.drop(drop_rows)

    drop_rows = df[df['Precipitacao'] == -999].index
    df = df.drop(drop_rows)

    drop_rows = df[df['RiscoFogo'] == -999].index
    df = df.drop(drop_rows)

    df = df.dropna(subset=['FRP', 'DiaSemChuva', 'Precipitacao', 'RiscoFogo'])


    df.to_csv('./data/treated_db/db_cerrado_cleaned.csv', index=False)
    return df

def check_errors_csv_list(df_list):
    clean_df_list = []

    for df in df_list:
        drop_rows = df[df['DiaSemChuva'] == -999].index
        df = df.drop(drop_rows)
        
        drop_rows = df[df['FRP'] == -999].index
        df = df.drop(drop_rows)

        drop_rows = df[df['Precipitacao'] == -999].index
        df = df.drop(drop_rows)

        drop_rows = df[df['RiscoFogo'] == -999].index
        df = df.drop(drop_rows)

        clean_df_list.append(df.dropna(subset=['FRP', 'DiaSemChuva', 'Precipitacao', 'RiscoFogo']))

    return clean_df_list
