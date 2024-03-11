import numpy as np
from collections import Counter
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest,f_classif



def palabras_mas_frecuentes_por_label(df, n,column_name):
    # Crear un diccionario para almacenar las palabras más frecuentes por label
    palabras_frecuentes = []

    # Obtener los valores únicos en la columna 'label'
    labels_unicos = df['label'].unique()

    # Iterar a través de los valores únicos en 'label'
    for label in labels_unicos:
        # Filtrar el DataFrame para obtener solo las filas con el valor de 'label' actual
        subconjunto = df[df['label'] == label]

        # Combinar todos los tokens en una lista
        tokens = [word for sublist in subconjunto[column_name] for word in sublist]

        # Contar la frecuencia de cada palabra en los tokens
        conteo_palabras = Counter(tokens)

        # Obtener las n palabras más frecuentes
        palabras_mas_frecuentes = conteo_palabras.most_common(n)

        # Agregar los resultados a la lista
        for palabra, frecuencia in palabras_mas_frecuentes:
            palabras_frecuentes.append((label, palabra, frecuencia))

    # Crear un DataFrame a partir de la lista de resultados
    df_resultados = pd.DataFrame(palabras_frecuentes, columns=['Label', 'Palabra', 'Frecuencia'])

    return df_resultados

def calculateLogOfOddsRatio(df):
    # Pivotaje y despivotaje
    tweets_pivot = df.groupby(["veracity","token"])["token"] \
                    .agg(["count"]).reset_index() \
                    .pivot(index = "token" , columns="veracity", values= "count")

    tweets_pivot = tweets_pivot.fillna(value=0)
    tweets_pivot.columns.name = None

    tweets_unpivot = tweets_pivot.melt(value_name='n', var_name='veracity', ignore_index=False)
    tweets_unpivot = tweets_unpivot.reset_index()

    # Selección de los autores elonmusk y mayoredlee
    tweets_unpivot = tweets_unpivot[tweets_unpivot.veracity.isin(['T', 'F'])]

    # Se añade el total de palabras de cada autor
    tweets_unpivot = tweets_unpivot.merge(
                        df.groupby('veracity')['token'].count().rename('N'),
                        how = 'left',
                        on  = 'veracity'
                    )

    # Cálculo de odds y log of odds de cada palabra
    tweets_logOdds = tweets_unpivot.copy()
    tweets_logOdds['odds'] = (tweets_logOdds.n + 1) / (tweets_logOdds.N + 1)
    tweets_logOdds = tweets_logOdds[['token', 'veracity', 'odds']] \
                        .pivot(index='token', columns='veracity', values='odds')
    tweets_logOdds.columns.name = None

    tweets_logOdds['log_odds'] = np.log(tweets_logOdds['T'] / tweets_logOdds['F'])
    tweets_logOdds['abs_log_odds'] = np.abs(tweets_logOdds.log_odds)

    # Si el logaritmo de odds es mayor que cero, significa que es una palabra con
    # mayor probabilidad de ser T. Esto es así porque el ratio sea ha
    # calculado como T/F.
    tweets_logOdds['veracidad_frecuente'] = np.where(tweets_logOdds.log_odds > 0,
                                              "T",
                                              "F"
                                    )
    return(tweets_logOdds)

def topWordsForNewsType(df):
    top_30 = calculateLogOfOddsRatio(df)[['log_odds', 'abs_log_odds', 'veracidad_frecuente']] \
        .groupby('veracidad_frecuente') \
        .apply(lambda x: x.nlargest(15, columns='abs_log_odds').reset_index()) \
        .reset_index(drop=True) \
        .sort_values('abs_log_odds', ascending=False)
    return(top_30[['veracidad_frecuente', 'token', 'log_odds', 'abs_log_odds']])

def palabras_por_label(df):
    palabras_T = []  # Lista para palabras con etiqueta 'T'
    palabras_F = []  # Lista para palabras con etiqueta 'F'

    # Iterar a través de las filas del DataFrame
    for index, row in df.iterrows():
        label = row['label']
        tokens = row['tokenized_content']

        if label == 'T':
            palabras_T.extend(tokens)
        elif label == 'F':
            palabras_F.extend(tokens)

    return palabras_T, palabras_F

def agregar_columnas_por_palabra(df, result):
    # Obtener las listas de palabras para etiquetas 'T' y 'F
    palabras_T, palabras_F = palabras_por_label(df)

    for word in result.keys():
        # Crea una nueva columna con el nombre de la palabra
        df[word] = 0

        # Itera a través de las filas del DataFrame
        for index, row in df.iterrows():
            label = row['label']
            if label == 'T' and word in palabras_T:
                df.at[index, word] = 1
            elif label == 'F' and word in palabras_F:
                df.at[index, word] = 1
    return df

# en este caso, el valor de las columnas (tokens) sera 1 o 0 si ese token se encuentra en el tweet.
def agregar_columnas_por_palabra2(df, result):
    # Recorre las palabras del diccionario
    for word in result.keys():
        # Crea una nueva columna con el nombre de la palabra
        df[word] = 0  # Inicializa con 0

        # Itera a través de las filas del DataFrame
        for index, row in df.iterrows():
            # Verifica si la palabra está en el array 'tokenized_content'
            if word in row['tokenized_content']:
                # Si la palabra está presente, asigna 1
                df.at[index, word] = 1

    return df

def columnas_a_eliminar(df, umbral):
    # Crear una lista de las columnas a eliminar
    columnas_a_eliminar = []

    # Itera a través de las columnas binarias (de la sexta en adelante)
    for columna in df.columns[5:]:
        # Calcula la frecuencia del valor 1 en la columna
        frecuencia_1 = (df[columna] == 1).sum() / len(df)
        
        # Calcula la frecuencia del valor 0 en la columna
        frecuencia_0 = (df[columna] == 0).sum() / len(df)
        
        # Si la frecuencia de 1 o 0 supera el umbral, agrega la columna a la lista de eliminación
        if frecuencia_1 > umbral or frecuencia_0 > umbral:
            columnas_a_eliminar.append(columna)

    return columnas_a_eliminar

def feature_selection_cascade(X_train, y_train, X_test, k_best, tree_threshold):
    # Paso 1: Selección de las k mejores características usando ANOVA F-statistic
    k_best_selector = SelectKBest(f_classif, k=k_best)
    X_train_k_best = k_best_selector.fit_transform(X_train, y_train)
    X_test_k_best = k_best_selector.transform(X_test)
    

    # Obtener las características seleccionadas después de la selección de las k mejores
    selected_features_indices_k_best = k_best_selector.get_support(indices=True)
    selected_features_names_k_best = X_train.columns[selected_features_indices_k_best]
    
    # Paso 2: Eliminación de características de baja varianza
    vt = VarianceThreshold(threshold=tree_threshold)
    X_train_vt = vt.fit_transform(X_train_k_best)
    X_test_vt = vt.transform(X_test_k_best)

    # Obtener las características seleccionadas después de la eliminación de baja varianza
    selected_features_indices_vt = vt.get_support(indices=True)
    selected_features_names_vt = selected_features_names_k_best[selected_features_indices_vt]
    print("\nCaracterísticas seleccionadas después de la eliminación de baja varianza:")
    print(selected_features_names_vt)

    return X_train_vt, X_test_vt

# Función para contar el número de palabras en cada array
def contar_palabras(arr):
    return len(arr)