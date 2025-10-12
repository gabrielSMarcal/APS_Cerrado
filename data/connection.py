import pandas as pd
import glob
import os

def merge_csv():
    csv_files_path = './data/db/*.csv' 

    all_csv_files = glob.glob(csv_files_path)

    list_of_dfs = []

    for file in all_csv_files:
        df = pd.read_csv(file)
        list_of_dfs.append(df)

    merged_df = pd.concat(list_of_dfs, ignore_index=True)

    return merged_df.to_csv('./db/db_cerrado.csv', index=False) 

def connection():
    try:

        file_path = './data/db/db_cerrado.csv'
        if not os.path.exists(file_path):
            merge_csv()

        df = pd.read_csv(file_path)

        return df
    except Exception as e:
        print(f'Falha ao conectar ao banco: {e}')
        return None


connection()
