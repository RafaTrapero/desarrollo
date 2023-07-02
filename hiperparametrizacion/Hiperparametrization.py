from sklearn.model_selection import GridSearchCV

def hiperParametrizacion(model,parameters,scoring,cv,X_train,Y_train,X_test,Y_test):

    # busqueda de hiperparametros del modelo
    grid_search = GridSearchCV(model, parameters, scoring=scoring, cv=cv,error_score='raise')
    grid_search.fit(X_train, Y_train)    

    # mejores parametros y puntuaciones
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_params,best_score


