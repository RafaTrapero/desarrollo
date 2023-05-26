import deeplake
import torch
import torchvision
import pandas as pd
import os

# cargamos los datos

os.chdir('D:/rafalete/TFG/datasets/LIAR')

df_train=pd.read_csv('train.tsv',sep='\t')

## añadimos un encabezado para diferenciar las columnas

df_train.columns = ['id','label','statement','subject','speaker','job_title','state_info','party_affiliation','barely_true_counts','false_counts','half_true_counts','mostly_true_counts','pants_onfire_counts','context']

## actualizamos el tsv
df_train.to_csv('train.tsv',sep='\t')

## filtramos solo aquellas afirmaciones que son tweets y volvemos a sobreescribir el tsv (el filtrado no es muy restrictivo)

df_train_filtrado=df_train.loc[ (df_train['statement'].str.contains('#',na=False)) | (df_train['statement'].str.contains('@',na=False)) | (df_train['context'].str.contains('tweet',na=False)) | (df_train['speaker'].str.contains('tweet',na=False)) | (df_train['statement'].str.contains('Twitter',na=False)) | (df_train['context'].str.contains('Twitter',na=False)) ]

dataloader = df_train_filtrado.pytorch(num_workers=0, batch_size=4, shuffle=False)


'''
for inputs, targets in dataloader:
    print("Inputs:", inputs)
    print("Targets:", targets)
    '''