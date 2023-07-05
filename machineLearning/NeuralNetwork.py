from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import tensorflow as tf


def neuralNetwork(X_train, Y_train, X_test, Y_test):
    # Crear el modelo
    model = Sequential()

    # Agregar capas al modelo
    model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compilar el modelo
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Vamos a convertir T y F a valores numéricos (1 y 0 respectivamente) y luego los pasamos a float
    Y_train[Y_train == "T"] = 1
    Y_train[Y_train == "F"] = 0
    Y_test[Y_test == "T"] = 1
    Y_test[Y_test == "F"] = 0
    Y_train = Y_train.astype(float)
    Y_test = Y_test.astype(float)

    # Convertir Y_train y Y_test a arreglos numpy
    Y_train = Y_train.values.astype(float)
    Y_test = Y_test.values.astype(float)

    # Convertir X_train a una matriz dispersa
    X_train_coo = coo_matrix(X_train)
    X_train_csr = X_train_coo.tocsr()

    # Convertir X_test a una matriz dispersa
    X_test_coo = coo_matrix(X_test)
    X_test_csr = X_test_coo.tocsr()

    # Entrenar el modelo
    model.fit(X_train_csr, Y_train, epochs=10, batch_size=32)

    # Evaluar el modelo en el conjunto de prueba
    loss, accuracy = model.evaluate(X_test_csr, Y_test)

    return accuracy



