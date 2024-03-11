
import re
import nltk 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict




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
    # Eliminación de espacios en blanco múltiples
    nuevo_texto = re.sub("\\s+", ' ', nuevo_texto)
    # Tokenización por palabras individuales
    nuevo_texto = nuevo_texto.split(sep = ' ')
    # Eliminación de tokens con una longitud < 2
    nuevo_texto = [token for token in nuevo_texto if len(token) > 1]
    
    return(nuevo_texto)



nltk.download('stopwords')

# Función para eliminar stopwords de una lista de palabras
def remove_stopwords(word_list):
    stop_words = set(stopwords.words('english'))
    return [word for word in word_list if word not in stop_words]

# Función para aplicar la eliminación de stopwords a una columna en el DataFrame
def remove_stopwords_from_column(df):
    df['tokenized_content'] = df['tokenized_content'].apply(remove_stopwords)
    return df

def top_words_by_label(df, n=10):
    # Obtén la lista de stopwords en inglés
    stop_words = set(stopwords.words('english'))

    # Aplana las listas en la columna 'tokenized_content' y divide en palabras individuales
    df['tokenized_content'] = df['tokenized_content'].apply(lambda x: ' '.join(x).split())

    # Filtra las stopwords y crea un DataFrame auxiliar
    df_filtered = df.explode('tokenized_content')
    df_filtered = df_filtered[~df_filtered['tokenized_content'].isin(stop_words)]

    # Cuenta la frecuencia de las palabras por label
    word_counts = df_filtered.groupby(['label', 'tokenized_content']).size().reset_index(name='Frecuencia de Aparición')

    # Ordena por frecuencia en orden descendente
    word_counts = word_counts.sort_values(by=['label', 'Frecuencia de Aparición'], ascending=[True, False])

    # Obtiene las palabras más frecuentes por label
    top_words = word_counts.groupby('label').head(n)

    return top_words

def get_unique_words_by_label(df):
    unique_words_by_label = {}  # Un diccionario para almacenar palabras únicas por etiqueta

    # Itera a través del DataFrame
    for index, row in df.iterrows():
        label = row['label']
        tokenized_content = row['tokenized_content']

        if label not in unique_words_by_label:
            unique_words_by_label[label] = set(tokenized_content)
        else:
            unique_words_by_label[label] = unique_words_by_label[label].difference(tokenized_content)

    # Crea una lista de palabras únicas por etiqueta
    unique_words_list = []
    for index, row in df.iterrows():
        label = row['label']
        unique_words_list.append(list(unique_words_by_label[label]))

    return unique_words_list    

def get_unique_words_by_label_menos_restrictiva(df):
    unique_words_by_label = {}  # Un diccionario para almacenar palabras únicas por etiqueta

    # Itera a través del DataFrame
    for index, row in df.iterrows():
        label = row['label']
        tokenized_content = row['tokenized_content']

        if label not in unique_words_by_label:
            unique_words_by_label[label] = set()  # Inicializa un conjunto vacío para la etiqueta

        unique_words_by_label[label].update(tokenized_content)  # Agrega todas las palabras tokenizadas a la etiqueta

    # Crea una lista de palabras únicas por etiqueta
    unique_words_list = []
    for index, row in df.iterrows():
        label = row['label']
        unique_words_list.append(list(unique_words_by_label[label]))

    return unique_words_list

    



def calculate_non_zero_tfidf(df):
    corpus = df['tokenized_content'].apply(lambda x: ' '.join(x))  # Convierte la lista de palabras en texto

    # Inicializa un vectorizador TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    
    # Ajusta el vectorizador al corpus completo
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # Obtén las características (palabras) y sus ponderaciones TF-IDF
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_values = tfidf_matrix.toarray()

    # Crea un diccionario de palabras y sus ponderaciones
    words_tfidf = {}
    for i, document in enumerate(corpus):
        for j, feature in enumerate(feature_names):
            tfidf = tfidf_values[i][j]
            if tfidf != 0.0:
                words_tfidf[feature] = tfidf

    return words_tfidf

def filter_tfidf_by_threshold(tfidf_dict, lower_threshold, upper_threshold):
    filtered_tfidf = {}

    for word, tfidf in tfidf_dict.items():
        if lower_threshold <= tfidf <= upper_threshold:
            filtered_tfidf[word] = tfidf

    return filtered_tfidf













