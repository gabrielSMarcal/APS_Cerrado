import pandas as pd
import glob
import os

from .check_data import check_errors, check_errors_csv_list
from .fonte import format_csv, format_csv_list

'''
Função para juntar os CSVs em um só
'''
def merge_csv(df_list):
    merged_df = pd.concat(df_list, ignore_index=True)

    return merged_df.to_csv('./data/treated_db/db_cerrado.csv', index=False) 

'''
Modulurização da função de pegar todos os CSVs de cada ano
'''
def get_df_list():
        CSVFILEPATH = './data/base_db/*.csv' 
        all_csv_files = glob.glob(CSVFILEPATH)
        all_csv_files.sort()
        df_list = []

        for file in all_csv_files:
            df_list.append(pd.read_csv(file))

        return df_list

'''
Gerar a conexão com o CSV tratado e formatado para o uso na aplicação
'''
def connection():
    try:
        DBPATH = './data/treated_db/db_cerrado.csv'
        if not os.path.exists(DBPATH):
            merge_csv(get_df_list())

        df = pd.read_csv(DBPATH)

        df = check_errors(df)
        df = format_csv(df)

        return df
    except Exception as e:
        print(f'Falha ao conectar ao banco: {e}')
        return None

'''
Gerar a conexão com a lista de CSVs tratados e formatados para o uso na aplicação
'''
def connection_list():
    try:
        CSVFILEPATH = './data/base_db/*.csv' 
        all_csv_files = glob.glob(CSVFILEPATH)
        all_csv_files.sort()
        df_list = []

        for file in all_csv_files:
            df_list.append(pd.read_csv(file))

        df_list = check_errors_csv_list(df_list)
        df_list = format_csv_list(df_list)

        return df_list
    except Exception as e:
        print(f'Falha ao conectar ao banco: {e}')
        return None
