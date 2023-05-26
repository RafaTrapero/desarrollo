import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


# Datos de ejemplo
texts = ['Me encantó esta película', 'No me gustó para nada', 'Gran actuación', 'Pésima calidad']

# Etiquetas correspondientes a los textos (1 para positivo, 0 para negativo)
labels = np.array([1, 0, 1, 0])

# Tokenización de los textos
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Obtención de la longitud máxima de secuencia
max_seq_length = max([len(seq) for seq in sequences])

# Padding de las secuencias para que todas tengan la misma longitud
sequences = pad_sequences(sequences, maxlen=max_seq_length)

# Creación del modelo
model = Sequential()
model.add(Embedding(1000, 32, input_length=max_seq_length))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

# Compilación del modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(sequences, labels, epochs=10, batch_size=1)

# Texto de prueba para la clasificación
test_text = ['Esta película es la mejor peli que he visto en mi vida']

# Preprocesamiento del texto de prueba
test_sequence = tokenizer.texts_to_sequences(test_text)
test_sequence = pad_sequences(test_sequence, maxlen=max_seq_length)

# Clasificación del texto de prueba
prediction = model.predict(test_sequence)

# Imprimir la predicción
print(prediction)
