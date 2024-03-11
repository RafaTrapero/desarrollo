import os
import pandas as pd


def process_dataset():
    os.chdir('C:/Users/Rafalete/Desktop/TFG/datasets/') #cambiar por ruta donde tengas tus datasets

    df=pd.read_csv('twitter_covid_labelled_mickey.csv')

    # Eliminiamos aquellas filas que esten etiquetadas con 'U' (unverified)
    df_filtrado = df[df['label'] != 'U']


    return df_filtrado