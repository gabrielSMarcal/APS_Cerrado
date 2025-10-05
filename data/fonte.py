import pandas as pd

def format_csv(csvpath):
    df = pd.read_csv(csvpath)
    df = df.drop(['Satelite', 'Pais', 'Bioma'], axis=1)
    df['DataHora'] = pd.to_datetime(df['DataHora']).dt.date

    return df

