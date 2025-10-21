from .check_data import check_errors, check_errors_csv_list
import pandas as pd


def format_csv():
    df = check_errors()
    df = df.drop(['Satelite', 'Pais', 'Bioma'], axis=1)
    df['DataHora'] = pd.to_datetime(df['DataHora'], errors='coerce')
    df['Data'] = df['DataHora'].dt.date
    df['RiscoFogo'] = df['RiscoFogo'] * 100

    return df

def format_csv_list():
    df_list = check_errors_csv_list()
    clean_df_list = []

    for df in df_list:
        clean_df = df
        clean_df = clean_df.drop(['Satelite', 'Pais', 'Bioma'], axis=1)
        clean_df['DataHora'] = pd.to_datetime(clean_df['DataHora'], errors='coerce')
        clean_df['Data'] = clean_df['DataHora'].dt.date
        clean_df['RiscoFogo'] = clean_df['RiscoFogo'] * 100

        clean_df_list.append(clean_df)
        
    return clean_df_list
