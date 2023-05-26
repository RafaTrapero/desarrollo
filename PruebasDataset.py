import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from transformers import BertTokenizer, TFBertModel
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.feature_extraction.text import CountVectorizer
import time
from sklearn.neural_network import MLPClassifier

inicio = time.time()


os.chdir('D:/rafalete/TFG/datasets/Pruebas/News _dataset')

df_fake=pd.read_csv('Fake.csv')
df_true=pd.read_csv('True.csv')



df_fake['veracity']='F'
df_true['veracity']='T'

## unimos ambos dataset en uno solo
df_global=pd.concat([df_true,df_fake],ignore_index=True)

## guardamos el nuevo df en csv
df_global.to_csv('df_global.csv',index=True)

## vemos el dataset
#print(df_global.shape)
#print(df_global.head())

# vemos cuantas T y F tenemos
#print(df_global.groupby('veracity').size())

####################### PREPROCESAMIENTO ##################




############################################################## Tecnicas  ##############################################################

# PRIMEROS PASOS COMUNES PARA TODOS LOS ALGORITMOS

X=df_global[['title','text','subject','date']]
Y=df_global['veracity'] # declaramos la variable de estudio

# aplicamos la tecnica de bolsa de palabras (Bag of Words) para procesar el texto de las columnas

''' 
vectorizer = CountVectorizer() # contamos la ocurrencia de cada termino en el texto y creamos una representacion vectorial en funcion al mismo
X_text = vectorizer.fit_transform(df_global['title'] + ' ' + df_global['text'])
'''

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2)) #min_df el numero minimo de documentos donde tienen que aparecer las palabras

X_text = tfidf.fit_transform(df_global['title'] + ' ' + df_global['text'])


# dividimos el conjunto en testing y training
X_train, X_test, y_train, y_test = train_test_split(X_text, Y, test_size=0.2, random_state=42)


### Algoritmos de clasificación

## **** DEEP LEARNING

# 1) CNN (Convolutional Neural Networks)


## *** MACHINE LEARNING ***

# 1) Decision Tree

decision_tree = DecisionTreeClassifier() #creamos el modelo 
decision_tree.fit(X_train, y_train) # entrenamos el modelo

# evaluamos la prediccion del modelo
y_pred = decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo DT: ", accuracy)

# 2) Random Forest

random_forest = RandomForestClassifier(n_estimators=100, random_state=42) #n_estimators es el numero de arboles que generamos y random_state es la semilla.
random_forest.fit(X_train, y_train)


# evaluamos la prediccion del modelo
y_pred = random_forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo RF: ", accuracy)



# 3) SVM
svm_model = SVC(kernel='linear') # kernel define cómo se mapearán los datos a un espacio de características de mayor dimensionalidad
svm_model.fit(X_train,y_train)

# evaluamos la prediccion del modelo
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo SVM: ", accuracy)

# 4) 


## VER TIEMPO DE EJECUCION
fin = time.time()
print("El tiempo de ejecución es:",fin-inicio)






