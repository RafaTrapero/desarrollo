from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from hiperparametrizacion.Hiperparametrization import hiperParametrizacion



def randomForest(X_train,Y_train,X_test,Y_test):

    #creamos el modelo
    dt=RandomForestClassifier()
    #seteamos los parametros para la hiperparametrizacion
    parameters = {
    'n_estimators': [100, 200, 300],  # Número de árboles en el bosque
    'max_depth': [None, 5, 10],       # Profundidad máxima de los árboles
    'min_samples_split': [2, 5, 10]   # Mínimo número de muestras requeridas para dividir un nodo
}


    #busqueda de parametros
    best_params = hiperParametrizacion(dt,parameters,"accuracy",5,X_train,Y_train,X_test,Y_test)[0]
    best_score =  hiperParametrizacion(dt,parameters,"accuracy",5,X_train,Y_train,X_test,Y_test)[1]

    #entrenamos y evaluamos el modelo
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X_train, Y_train)
    accuracy = best_model.score(X_test, Y_test)

    return accuracy