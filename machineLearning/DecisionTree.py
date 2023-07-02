from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from hiperparametrizacion.Hiperparametrization import hiperParametrizacion



def decisionTree(X_train,Y_train,X_test,Y_test):

    #creamos el modelo
    dt=DecisionTreeClassifier()
    #seteamos los parametros para la hiperparametrizacion
    parameters = {
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
    }

    #busqueda de parametros
    best_params = hiperParametrizacion(dt,parameters,"accuracy",5,X_train,Y_train,X_test,Y_test)[0]
    best_score =  hiperParametrizacion(dt,parameters,"accuracy",5,X_train,Y_train,X_test,Y_test)[1]

    #entrenamos y evaluamos el modelo
    best_model = DecisionTreeClassifier(**best_params)
    best_model.fit(X_train, Y_train)
    accuracy = best_model.score(X_test, Y_test)

    return accuracy