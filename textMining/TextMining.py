
import re
import nltk 
from nltk.corpus import stopwords





def cleanAndtokenize(texto):
    '''
    Esta función limpia y tokeniza el texto en palabras individuales.
    El orden en el que se va limpiando el texto no es arbitrario.
    '''
    # Se convierte todo el texto a minúsculas
    nuevo_texto = texto.lower()
    # Eliminación de páginas web (palabras que empiezan por "http")
    nuevo_texto = re.sub('http\S+', ' ', nuevo_texto)
    # Eliminación de signos de puntuación
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    nuevo_texto = re.sub(regex , ' ', nuevo_texto)
    # Eliminación de números
    nuevo_texto = re.sub("\d+", ' ', nuevo_texto)
    # Eliminación de espacios en blanco múltiples
    nuevo_texto = re.sub("\\s+", ' ', nuevo_texto)
    # Tokenización por palabras individuales
    nuevo_texto = nuevo_texto.split(sep = ' ')
    # Eliminación de tokens con una longitud < 2
    nuevo_texto = [token for token in nuevo_texto if len(token) > 1]
    
    return(nuevo_texto)


def unnset(df):
    '''
    Esta funcion sirve para  obligar que se cumpla la funcion de tidy data (una observacion por fila)
    Para ello ducplicamos el valor del resto de filas
    '''
    df_tidy = df.explode(column='tokenized_title')
    df_tidy = df_tidy.drop(columns='text')
    df_tidy = df_tidy.rename(columns={'tokenized_title':'token'})  

    return(df_tidy)





def stopwordFiltering(df):

    stop_words = list(stopwords.words('english'))
    df = df[~(df["token"].isin(stop_words))] # filtra el DataFrame original df y guarda en df solo las filas donde los valores de la columna "token" no están en la lista de palabras stop_words.
    return(df)








