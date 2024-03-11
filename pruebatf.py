from sklearn.model_selection import train_test_split
from procesamientoDataset.ProcessDataset import process_dataset
import pandas as pd
import tensorflow_hub as hub 


df = process_dataset()
df=df.drop(columns=['No.','source','sentiment','reply numbers','retweet numbers','likes numbers'])
#df.to_csv('df_tf.tsv',sep='\t',index=True)

##################### DESBALANCEO (DOWN SAMPLING) Y PREPROCESAMIENTO #####################

#print(df.groupby('label').describe()) ## vemos que hay un desbalanceo (F:540 y T:1040)

df_true=df[df['label']=='T']

df_false=df[df['label']=='F']

df_true_downsampled=df_true.sample(df_false.shape[0])

df_balanced=pd.concat([df_true_downsampled,df_false])

print(df_balanced['label'].value_counts())

df_balanced['label'] = df_balanced['label'].replace({'F': 0, 'T': 1})

X_train, X_test, y_train, y_test = train_test_split(df_balanced['content'],df_balanced['label'], stratify=df_balanced['label'])

##################### BERT #####################
bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")