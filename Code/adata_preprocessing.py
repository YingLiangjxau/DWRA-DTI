import logging
import subprocess as sp
from tqdm import tqdm
datasets_o = ['DrugBank','BIOSNAP', 'BindingDB', 'Davis', 'E', 'GPCR', 'IC', 'NR']

for database in datasets_o:
    try:
        print(f'database_preprocessing {database}')
        return_code = sp.check_call(['python', 'Code/data_preprocessing.py', '-d', f'{database}'])
        if return_code == 0:
            print(f'EXIT CODE 0 FOR {database.upper()}')
    except sp.CalledProcessError as e:
        logging.info(e.output)