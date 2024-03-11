from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict,StratifiedKFold


def logistic_regression_hyperparameter_tuning_cv(X_train, y_train, X_test, y_test, cv):
    # Definir el modelo de regresión logística
    model = LogisticRegression(random_state=42,C=0.000001)

    # Definir la cuadrícula de hiperparámetros a explorar
    param_grid = {
        'penalty': ['l2'],  # Usar solo 'l2' para solver 'lbfgs'
        'solver': ['liblinear'],  # Usar 'lbfgs' como solver
        'max_iter': [800, 900, 1000]  # Ajustar el número máximo de iteraciones
    }

    # Inicializar GridSearchCV con validación cruzada
    stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=stratified_cv, scoring='balanced_accuracy', error_score='raise')


    # Realizar la búsqueda de hiperparámetros en el conjunto de entrenamiento
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo
    best_model = grid_search.best_estimator_

    # Hacer predicciones en el conjunto de prueba
    y_pred = best_model.predict(X_test)

    # Evaluar el rendimiento del mejor modelo
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f'Mejores hiperparámetros: {grid_search.best_params_}')
    print(f'Accuracy con mejores hiperparámetros: {accuracy:.4f}')
    print('\nConfusion Matrix:')
    print(confusion_mat)
    print('\nClassification Report:')
    print(classification_rep)

    return best_model

def logistic_regression_default(X_train, y_train, X_test, y_test):
    # Definir el modelo de regresión logística con valores predeterminados
    model = LogisticRegression(random_state=42,C=0.0001)

    # Entrenar el modelo con los datos de entrenamiento
    model.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Evaluar el rendimiento del modelo
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f'Accuracy con hiperparámetros predeterminados: {accuracy:.4f}')
    print('\nConfusion Matrix:')
    print(confusion_mat)
    print('\nClassification Report:')
    print(classification_rep)

    return model

def logistic_regression_tuning_cv(X_train, y_train, X_test, y_test, cv):
    # Definir el modelo de regresión logística
    model = LogisticRegression(random_state=42, C=0.1)

    # Realizar predicciones con validación cruzada en el conjunto de entrenamiento
    y_pred_cv = cross_val_predict(model, X_train, y_train,cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42))

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

