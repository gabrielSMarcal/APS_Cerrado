import pandas as pd
import glob
import os

def merge_csv(df_list):
    merged_df = pd.concat(df_list, ignore_index=True)

    return merged_df.to_csv('./data/treated_db/db_cerrado.csv', index=False) 

def get_df_list():
        CSVFILEPATH = './data/base_db/*.csv' 
        all_csv_files = glob.glob(CSVFILEPATH)
        all_csv_files.sort()
        df_list = []

        for file in all_csv_files:
            df_list.append(pd.read_csv(file))
        
        return df_list

def connection():
    try:
        DBPATH = './data/treated_db/db_cerrado.csv'
        if not os.path.exists(DBPATH):
            merge_csv(get_df_list())

        df = pd.read_csv(DBPATH)

        return df
    except Exception as e:
        print(f'Falha ao conectar ao banco: {e}')
        return None
