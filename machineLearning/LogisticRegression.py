from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 3) Logistic Regression

def logisticRegression(X_train,Y_train,X_test,Y_Test):

    lr=LogisticRegression()
    lr.fit(X_train,Y_train) 

    # evaluamos la prediccion del modelo
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(Y_Test, y_pred)
    return accuracy
