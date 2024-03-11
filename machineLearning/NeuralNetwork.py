import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import tensorflow as tf

def neural_network_tuning_cv(X_train, y_train, X_test, y_test, cv):
    # Definir el modelo de red neuronal
    model = MLPClassifier(random_state=42, hidden_layer_sizes=(300,), max_iter=500)

    # Realizar predicciones con validación cruzada en el conjunto de entrenamiento
    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42))

    # Entrenar el modelo en todo el conjunto de entrenamiento
    model.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred_test = model.predict(X_test)

    # Evaluar el rendimiento del modelo en el conjunto de prueba
    accuracy_test = accuracy_score(y_test, y_pred_test)
    confusion_mat_test = confusion_matrix(y_test, y_pred_test)
    classification_rep_test = classification_report(y_test, y_pred_test)

    print(f'Accuracy en conjunto de prueba: {accuracy_test:.4f}')
    print('\nConfusion Matrix en conjunto de prueba:')
    print(confusion_mat_test)
    print('\nClassification Report en conjunto de prueba:')
    print(classification_rep_test)

    # Imprimir el resultado de la validación cruzada en el conjunto de entrenamiento
    accuracy_cv = accuracy_score(y_train, y_pred_cv)
    confusion_mat_cv = confusion_matrix(y_train, y_pred_cv)
    classification_rep_cv = classification_report(y_train, y_pred_cv)

    print(f'\nAccuracy promedio en validación cruzada en conjunto de entrenamiento: {accuracy_cv:.4f}')
    print('\nConfusion Matrix en validación cruzada en conjunto de entrenamiento:')
    print(confusion_mat_cv)
    print('\nClassification Report en validación cruzada en conjunto de entrenamiento:')
    print(classification_rep_cv)

    return model

def neural_network_tuning_cv_tensorflow(X_train, y_train, X_test, y_test, cv):
    # Función para crear el modelo de red neuronal
    def create_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(100, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    # Listas para almacenar las métricas de evaluación
    accuracy_cv_scores = []
    confusion_matrices_cv = []

    # Validación cruzada
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Crear el modelo
        model = create_model()

        # Entrenar el modelo
        model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)

        # Evaluar el modelo en el conjunto de validación
        _, accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        accuracy_cv_scores.append(accuracy)

        # Hacer predicciones en el conjunto de validación
        y_pred_val = np.round(model.predict(X_val_fold)).flatten()

        # Calcular la matriz de confusión
        confusion_matrices_cv.append(confusion_matrix(y_val_fold, y_pred_val))

    # Promedio de las métricas de evaluación de la validación cruzada
    # Promedio de las métricas de evaluación de la validación cruzadaSS
        mean_accuracy_cv = np.mean(accuracy_cv_scores, axis=0)
        mean_confusion_matrix_cv = np.mean(confusion_matrices_cv, axis=0, dtype=np.float64)


    # Entrenar el modelo en todo el conjunto de entrenamiento con los mejores hiperparámetros encontrados
    final_model = create_model()
    final_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Evaluar el modelo en el conjunto de prueba
    _, accuracy_test = final_model.evaluate(X_test, y_test, verbose=0)

    # Hacer predicciones en el conjunto de prueba
    y_pred_test = np.round(final_model.predict(X_test)).flatten()

    # Calcular la matriz de confusión en el conjunto de prueba
    confusion_mat_test = confusion_matrix(y_test, y_pred_test)
    classification_rep_test = classification_report(y_test, y_pred_test)

    print(f'Accuracy en conjunto de prueba: {accuracy_test:.4f}')
    print('\nConfusion Matrix en conjunto de prueba:')
    print(confusion_mat_test)
    print('\nClassification Report en conjunto de prueba:')
    print(classification_rep_test)

    # Imprimir el resultado de la validación cruzada en el conjunto de entrenamiento
    print(f'\nAccuracy promedio en validación cruzada en conjunto de entrenamiento: {mean_accuracy_cv:.4f}')
    print('\nConfusion Matrix promedio en validación cruzada en conjunto de entrenamiento:')
    print(mean_confusion_matrix_cv)

    return final_model
