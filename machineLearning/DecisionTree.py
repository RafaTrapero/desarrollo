from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from hiperparametrizacion.Hiperparametrization import hiperParametrizacion



from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def decision_tree_tuning_cv(X_train, y_train, X_test, y_test, cv):
    # Definir el modelo de árbol de decisión
    dt_classifier = DecisionTreeClassifier(random_state=42)

    # Definir los hiperparámetros para la búsqueda en cuadrícula
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 1, 2, 3],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Realizar búsqueda en cuadrícula para encontrar los mejores hiperparámetros
    grid_search = GridSearchCV(dt_classifier, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo después de la búsqueda en cuadrícula
    best_dt_model = grid_search.best_estimator_

    # Realizar predicciones con validación cruzada en el conjunto de entrenamiento
    y_pred_cv = cross_val_predict(best_dt_model, X_train, y_train, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42))

    # Entrenar el modelo en todo el conjunto de entrenamiento
    best_dt_model.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred_test = best_dt_model.predict(X_test)

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

    return best_dt_model

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def decision_tree_cv(X_train, y_train, X_test, y_test, cv):
    # Definir el modelo de árbol de decisión
    dt_classifier = DecisionTreeClassifier(random_state=42)

    # Realizar validación cruzada para encontrar el rendimiento promedio en el conjunto de entrenamiento
    y_pred_cv = cross_val_predict(dt_classifier, X_train, y_train, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42))

    # Entrenar el modelo en todo el conjunto de entrenamiento
    dt_classifier.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred_test = dt_classifier.predict(X_test)

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

    return dt_classifier

def decision_tree(X_train, y_train, X_test, y_test):
    # Definir el modelo de árbol de decisión
    dt_classifier = DecisionTreeClassifier(random_state=42)

    # Entrenar el modelo en todo el conjunto de entrenamiento
    dt_classifier.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred_test = dt_classifier.predict(X_test)

    # Evaluar el rendimiento del modelo en el conjunto de prueba
    accuracy_test = accuracy_score(y_test, y_pred_test)
    confusion_mat_test = confusion_matrix(y_test, y_pred_test)
    classification_rep_test = classification_report(y_test, y_pred_test)

    print(f'Accuracy en conjunto de prueba: {accuracy_test:.4f}')
    print('\nConfusion Matrix en conjunto de prueba:')
    print(confusion_mat_test)
    print('\nClassification Report en conjunto de prueba:')
    print(classification_rep_test)

    return dt_classifier
