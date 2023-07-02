from procesamientoDataset.ProcessDataset import process_dataset
from textMining.TextMining import cleanAndtokenize,unnset,stopwordFiltering
from utils.Utils import mostUsedWordsByVeracity,topWordsForNewsType
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from machineLearning.LogisticRegression import logisticRegression
from machineLearning.DecisionTree import decisionTree
from machineLearning.RandomForest import randomForest

## PROCESAMIENTO DATASET   
df_global = process_dataset()
df_global2 = process_dataset()

# desde aqui hasta la linea 54 es analisis y ejemplos de lo que vamos a aplicar

## TEXT MINING

# Se aplica la funcion de limpieza y tokenización a cada noticia (columna 'title')
df_global['tokenized_title'] = df_global['title'].apply(lambda x: cleanAndtokenize(x))

# Hacemos el unset
df_global=unnset(df_global)
df_global.to_csv('df_global_unset.csv', index=True)

# Vemos las palabras más utilizadas por veracidad
print("================== PALABRAS MAS USADAS POR VERACIDAD ==================")
print(mostUsedWordsByVeracity(df_global))

# Podemos ver que los términos mas usados son articulos, preposiciones. Para ello eliminamos las stopwords.
df_global_sin_stopwords=stopwordFiltering(df_global)


# Top 5 palabras por veracidad (sin stopwords)
print("================== PALABRAS MAS USADAS POR VERACIDAD (SIN STOPWORDS) ==================")

print(mostUsedWordsByVeracity(df_global_sin_stopwords))


'''
Podemos ver que hay palabras como 'trump' que se encuentran tanto en las noticias fake, como en las verdaderas. 
Para ello hacemos una comparación para ver que palabras aparecen mucho en las fake news y no en las verdaderas,
y viceversa.
'''

# # Comparacion en el uso de palabras

# '''
# Se estudia que palabras se utilizan de forma mas diferenciada por cada usuario 
# mediante el log of odds ratio
# '''
print("================== TOP 15 PALABRAS QUE MAS APARECEN POR TIPO DE NOTICIA ==================")

print(topWordsForNewsType(df_global_sin_stopwords))

# TF-IDF

'''
Empleando los tweets de entrenamiento se crea un matriz tf-idf en la que cada columna es un término, 
cada fila un documento y el valor de intersección el tf-idf correspondiente. Esta matriz representa el espacio n-dimensional en el que se proyecta cada tweet.
'''
# Creación de la matriz tf-idf
# ==============================================================================

stop_words = list(stopwords.words('english'))
# tfidf_vectorizador = TfidfVectorizer(
#                         tokenizer  = cleanAndtokenize,
#                         min_df     = 1,
#                         max_df=5, 
#                         stop_words = stop_words,
#                         ngram_range=(1,2)
#                     )

# PREPARAMOS LOS DATOS PARA PODER USARLOS EN LOS DISTINTOS ALGORITMOS


X=df_global2[['title','text','subject','date']]
Y=df_global2['veracity'] # declaramos la variable de estudio

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2),tokenizer=cleanAndtokenize,stop_words=stop_words) #min_df el numero minimo de documentos donde tienen que aparecer las palabras // ngran_range=el numero de gramas (mono gramas y bigramas)

X_text = tfidf.fit_transform(df_global2['title'])


print("Dimensiones de X:", X.shape)
print("Dimensiones de  Y: ", Y.shape)

print("Dimensiones de X_text:", X_text.shape)
print("Dimensiones de  Y: ", Y.shape)
# vemos que el numero de dimensiones de X_text no es el mismo que en Y. Por ello ejecutamos el siguiente codigo YA ME FUNCIONAAAAAAAAAAAAAAAAAAAAAAAAAAAA


# dividimos el conjunto en testing y training
X_train, X_test, y_train, y_test = train_test_split(X_text, Y, test_size=0.2, random_state=42)


# Seguimos con tf-idf
# tfidf_vectorizador.fit(X_train)

# tfidf_train = tfidf_vectorizador.transform(X_train)
# tfidf_test  = tfidf_vectorizador.transform(X_test)

# MACHINE LEARNING

# 1) Logistic Regression
#print("Precisión del modelo LR: ", logisticRegression(X_train,y_train,X_test,y_test))

# 2) Decision Tree
#print("Precision del modelo DT: ",decisionTree(X_train,y_train,X_test,y_test))

# 3) Random Forest
print("Precision del modelo DT: ",randomForest(X_train,y_train,X_test,y_test))





