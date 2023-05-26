# imports
import deeplake
import pandas as pd
import os
import re # regular expresions

# 1) Cargamos y preprocesamos los datasets

## cargamos el directorio donde los tenemos descargados

os.chdir('D:/rafalete/TFG/datasets/LIAR')

df_train=pd.read_csv('train.tsv',sep='\t')
df_test=pd.read_csv('test.tsv',sep='\t')
df_validation=pd.read_csv('valid.tsv',sep='\t')

## añadimos un encabezado para diferenciar las columnas

df_train.columns = ['id','label','statement','subject','speaker','job_title','state_info','party_affiliation','barely_true_counts','false_counts','half_true_counts','mostly_true_counts','pants_onfire_counts','context']

## actualizamos el tsv
df_train.to_csv('train.tsv',sep='\t')

## filtramos solo aquellas afirmaciones que son tweets y volvemos a sobreescribir el tsv (el filtrado no es muy restrictivo)

df_train_filtrado=df_train.loc[ (df_train['statement'].str.contains('#',na=False)) | (df_train['statement'].str.contains('@',na=False)) | (df_train['context'].str.contains('tweet',na=False)) | (df_train['speaker'].str.contains('tweet',na=False)) | (df_train['statement'].str.contains('Twitter',na=False)) | (df_train['context'].str.contains('Twitter',na=False)) ]
df_train_filtrado.to_csv('train2.tsv',sep='\t')

# 2) Creamos los datasets con la info de los tweets

## creamos una lista con los ids
id_list = df_train_filtrado['id'].tolist()

## extraemos unicamente el nombre eliminando la extension 
id_list_final=[]
for id in id_list:
    id_final=re.search(r'\d+', id).group()
    id_list_final.append(id_final)

## extraemos el json de cada afirmacion
for id in id_list_final:
    url = f"http://www.politifact.com//api/v/2/statement/{id}/?format=json"
    os.system(f"wget {url}")





