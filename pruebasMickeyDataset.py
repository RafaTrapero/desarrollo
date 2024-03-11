import os
import pandas as pd

os.chdir('C:/Users/Rafalete/Desktop/TFG/datasets/')
df=pd.read_csv('twitter_covid_labelled_mickey.csv')
filtered_df = df[df['label'] == 'U']

# Contar las filas que cumplen con el criterio
count = filtered_df.shape[0]
print(df.shape[0]-count)