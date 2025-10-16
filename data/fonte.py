from .check_data import check_errors
import pandas as pd

def format_csv():
    df, list_of_dfs = check_errors()
    df = df.drop(['Satelite', 'Pais', 'Bioma'], axis=1)
    df['DataHora'] = pd.to_datetime(df['DataHora'], errors='coerce')
    df['Data'] = df['DataHora'].dt.date
    df['RiscoFogo'] = df['RiscoFogo'] * 100

    clean_df_list = []

    for dfs in list_of_dfs:
        clean_df = dfs
        clean_df = clean_df.drop(['Satelite', 'Pais', 'Bioma'], axis=1)
        clean_df['DataHora'] = pd.to_datetime(clean_df['DataHora'], errors='coerce')
        clean_df['Data'] = clean_df['DataHora'].dt.date
        clean_df['RiscoFogo'] = clean_df['RiscoFogo'] * 100

        clean_df_list.append(clean_df)

    return df, clean_df_list
