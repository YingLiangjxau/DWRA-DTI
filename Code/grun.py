import logging
import subprocess as sp


from tqdm import tqdm


database_o = ['DrugBank', 'BIOSNAP', 'BindingDB', 'Davis', 'E', 'GPCR', 'IC', 'NR']
database_t = database_o

nreps_training = 5
nreps = 5

for rep_t in tqdm(range(nreps_training), desc='Repetitions'):

    for database1 in database_o:

        print('======================')
        print(f'Working with {database1}')

        for database2 in database_t:
            if database2 != database1:
                database = f'{database1}' + '_WO_' + f'{database2}'
                print(database)

                try:
                    print(f'database {database}')
                    return_code = sp.check_call(['python', 'Code/main.py', '-d', f'{database}'])
                    if return_code == 0:
                        print(f'EXIT CODE 0 FOR {database.upper()}')

                except sp.CalledProcessError as e:
                    logging.info(e.output)

            else:
                database = f'{database1}'
                print(database)

                try:
                    print(f'database {database}')
                    return_code = sp.check_call(
                        ['python', 'Code/main.py', '-d', f'{database}'])
                    if return_code == 0:
                        print(f'EXIT CODE 0 FOR {database.upper()}')

                except sp.CalledProcessError as e:
                    logging.info(e.output)

        print(f'Finished with {database1}')


# database_o = ['DrugBank', 'BIOSNAP', 'BindingDB', 'Davis', 'E', 'GPCR', 'IC', 'NR']
# database_1='NR'
# for i in range(5):
#     for database_2 in database_o:
#         if database_2 != database_1:
#             database = f'{database_1}' + '_WO_' + f'{database_2}'
#             try:
#                 print(f'dataset: {database},hidden_channels 18')
#                 return_code=sp.check_call(['python', 'Code/main.py', '-d', f'{database}'])
#                 if return_code == 0:
#                     print(f'EXIT CODE 0 FOR {database.upper()} with hd 18')
#             except sp.CalledProcessError as e:
#                 logging.info(e.output)




