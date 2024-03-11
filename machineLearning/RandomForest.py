from sklearn.calibration import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from hiperparametrizacion.Hiperparametrization import hiperParametrizacion
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report




def randomForest(X_train,Y_train,X_test,Y_test):

    #creamos el modelo
    rf=RandomForestClassifier()
    #seteamos los parametros para la hiperparametrizacion
    parameters = {
    'n_estimators': [10,50,100],  # Número de árboles en el bosque
    'max_depth': [None, 5, 10],       # Profundidad máxima de los árboles
    'min_samples_split': [2, 5, 10]   # Mínimo número de muestras requeridas para dividir un nodo
}


    #busqueda de parametros
    best_params = hiperParametrizacion(rf,parameters,"accuracy",5,X_train,Y_train,X_test,Y_test)[0]
    best_score =  hiperParametrizacion(rf,parameters,"accuracy",5,X_train,Y_train,X_test,Y_test)[1]

    #entrenamos y evaluamos el modelo
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X_train, Y_train)
    accuracy = best_model.score(X_test, Y_test)

    return accuracy

def random_forest_hyperparameter_tuning_cv(X_train, y_train, X_test, y_test, cv):
    # Definir el modelo de Random Forest
    model = RandomForestClassifier(random_state=42)

    # Definir la cuadrícula de hiperparámetros a explorar
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Inicializar GridSearchCV con validación cruzada
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

def random_forest_tuning_cv(X_train, y_train, X_test, y_test, cv):
    # Definir el modelo de Random Forest
    model = RandomForestClassifier(random_state=42)

    # Definir la cuadrícula de hiperparámetros a ajustar
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, 40, 50],
        'class_weight': ['balanced']
    }

    # Inicializar GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2, n_jobs=-1)

    # Realizar predicciones con validación cruzada en el conjunto de entrenamiento
    y_pred_cv = cross_val_predict(grid_search, X_train, y_train, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42))

    # Entrenar el modelo en todo el conjunto de entrenamiento con los mejores hiperparámetros encontrados
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo encontrado por GridSearchCV
    best_model = grid_search.best_estimator_

    # Hacer predicciones en el conjunto de prueba
    y_pred_test = best_model.predict(X_test)

    # Evaluar el rendimiento del modelo en el conjunto de prueba
    accuracy_test = accuracy_score(y_test, y_pred_test)
    confusion_mat_test = confusion_matrix(y_test, y_pred_test)
    classification_rep_test = classification_report(y_test, y_pred_test)

    print(f'Accuracy en conjunto de prueba: {accuracy_test:.4f}')
    print('\nConfusion Matrix en conjunto de prueba:')
    print(confusion_mat_test)
    print('\nClassification Report en conjunto de prueba:')
    print(classification_rep_test)

    # Imprimir los mejores hiperparámetros encontrados
    print('\nMejores hiperparámetros encontrados:')
    print(grid_search.best_params_)

    # Imprimir el resultado de la validación cruzada en el conjunto de entrenamiento
    accuracy_cv = accuracy_score(y_train, y_pred_cv)
    confusion_mat_cv = confusion_matrix(y_train, y_pred_cv)
    classification_rep_cv = classification_report(y_train, y_pred_cv)

    print(f'\nAccuracy promedio en validación cruzada en conjunto de entrenamiento: {accuracy_cv:.4f}')
    print('\nConfusion Matrix en validación cruzada en conjunto de entrenamiento:')
    print(confusion_mat_cv)
    print('\nClassification Report en validación cruzada en conjunto de entrenamiento:')
    print(classification_rep_cv)

    return best_model
