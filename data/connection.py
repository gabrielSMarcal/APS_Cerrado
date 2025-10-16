import pandas as pd
import glob
import os

def merge_csv(list_of_dfs):
    merged_df = pd.concat(list_of_dfs, ignore_index=True)

    return merged_df.to_csv('./data/treated_db/db_cerrado.csv', index=False) 

def connection():
    try:
        csv_files_path = './data/base_db/*.csv' 
        
        all_csv_files = glob.glob(csv_files_path)
        
        list_of_dfs = []

        for file in all_csv_files:
            list_of_dfs.append(pd.read_csv(file))

        file_path = './data/treated_db/db_cerrado.csv'
        if not os.path.exists(file_path):
            merge_csv(list_of_dfs)

        df = pd.read_csv(file_path)

        return df, list_of_dfs
    except Exception as e:
        print(f'Falha ao conectar ao banco: {e}')
        return None
