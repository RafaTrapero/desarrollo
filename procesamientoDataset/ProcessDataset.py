import os
import pandas as pd


def process_dataset():
    os.chdir('D:/rafalete/TFG/datasets/Pruebas/Newsdataset/')

    df_fake = pd.read_csv('Fake.csv')
    df_true = pd.read_csv('True.csv')

    df_fake['veracity'] = 'F'
    df_true['veracity'] = 'T'

    # Unir ambos datasets en uno solo
    df_global = pd.concat([df_true, df_fake], ignore_index=True)

    # Desordenamos las filas del dataframe
    df_global=df_global.sample(frac=1)

    # Guardar el nuevo df en csv
    df_global.to_csv('df_global.csv', index=True)

    return df_global