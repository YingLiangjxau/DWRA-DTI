import logging
import subprocess as sp


from tqdm import tqdm
nreps = 20
datasets_o = ['DrugBank','BIOSNAP', 'BindingDB', 'Davis', 'E', 'GPCR', 'IC', 'NR']
# datasets_o = ['DrugBank','BIOSNAP', 'BindingDB']
# database=datasets_o[2]
for i in range(nreps):
    for database in datasets_o:
        try:
            print(f'database {database}, hidden_channels 18')
            return_code = sp.check_call(['python', 'Code/main.py', '-d', f'{database}'])
            if return_code == 0:
                print(f'EXIT CODE 0 FOR {database.upper()} with hd 18')

        except sp.CalledProcessError as e:
            logging.info(e.output)

