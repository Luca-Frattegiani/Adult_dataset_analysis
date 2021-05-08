import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def display_coefficients(intercept, coefficients, df):
    """
    Function  that shows the graphic representation of the coefficients estimated by a Logistic Regressor through the use
    of a barplot.
    
    PARAMETERS:
    1) intercetta = Array containing the intercept of the model.
    2) coefficienti = Array containing the coefficients of the model.
    3) df = Dataframe used for the analysis.
    
    OUTPUT:
    Graphic representation.
    """
    names = ['intercept']

    for column in df.columns[:len(df.columns)-1]:
        names.append(column)
    
    parameters = np.concatenate((intercept, coefficients.flatten()))
    
    fig = plt.figure(figsize = (10, 6))
    plt.barh(width = parameters[::-1], y = names[::-1])
    plt.show()

def metrics(y_test, y_predict, assign = False, show = True):
    """
    Function to print and compute che main metrics of a Classification Model following this order: accuracy,
    precision, recall, f1.
    
    PARAMETERS:
    1) y_test = Array with the true values observed in the test set.
    2) y_predict = Array with the values predicted by the model on the same test set (X_test).
    3) assegna = Boolean value, if "True" compute the metrics (default "False").
    4) stampa = Boolean value, if "True" print the obtained results (default = "True").
    
    OUTPUT:
    If stampa = "True": Print representation.
    2) assegna = True: Float values of the metrics.
    """
    cm = confusion_matrix(y_test, y_predict)
    precision = (cm[1,1])/((cm[1,1])+(cm[0,1]))
    recall = (cm[1,1])/((cm[1,1])+(cm[1,0]))
    f1 = (2 * recall * precision) / (recall + precision)
    accuracy = (cm[0,0] + cm[1,1])/(cm.sum())
    
    if show == True:
        lista = [accuracy, precision, recall, f1]
        lista2 = ["Accuracy", "Precision", "Recall", "f1"]
        for metric, name in zip(lista, lista2):
            print(f"{name}: {metric}")
            print("-"*114)
    
    if assign == True:
        return accuracy, precision, recall, f1

def insert (models, y_test, y_predict, name):
    """
    Function to insert the metrics estimated for a model into a list through the use of a dictionary.
    
    PARAMETERS:
    
    1) modelli = List containing the estimated model.
    2) y_test = Array with the true values observed in the test set.
    3) y_predict = Array with the values predicted by the model on the same test set (X_test).
    4) nome = String referred to the name of the model
    
    OUTPUT:
    None (the function only insert the model performance into a list)
    """
    accuracy, precision, recall, f1 = metrics(y_test, y_predict, assign = True, show = False)
    
    name += ":"
    
    models.append({
        "Name" : name,
        "Accuracy" : accuracy,
        "Precision" : precision,
        "Recall" : recall,
        "f1" : f1})

def report(models):
    """
    Function which provides a description of the models estimated printing their metrics.
    
    PARAMETERS:
    modelli = List containing the estimated model.
    
    OUTPUT:
    Print of the model's metrics to compare the performance.
    """
    
    count = 0
    for model in models:
        for key in model.keys():
            if key == "Name":
                print(f"{count+1}) {model[key]}\n")
                if count != 0:
                    print("\t\t\t\tVariation from the previous model:")
            else:
                if count != 0:
                    if key == "f1":
                        print(f"{key}: {model[key]}\t\t({model[key] - previous[key]})")
                    else:
                        print(f"{key}: {model[key]}\t({model[key] - previous[key]})")                        
                else:
                    print(f"{key}: {model[key]}")                    
        print("-"*114)
        previous = model
        count += 1

def show_features_importances(importances, names):
    """
    Function which shows the feature importances for a DecisionTree through the use of a barplot.
    
    PARAMETERS:
    importances = Array containing the attribute 'features_importances' of a DecisionTree.
    
    nomi = List containing the name of the predictors used in the model.
    
    OUTPUT:
    Graphic reperesentation.
    """    
    normalized = (importances/importances.max())*100
    percentage = (importances*100)
    
    dn = pd.DataFrame(data = {"feature_importances": normalized}, index = names)
    dp = pd.DataFrame(data = {"feature_importances": percentage}, index = names)
    dn.sort_values(by = "feature_importances", inplace = True)
    dp.sort_values(by = "feature_importances", inplace = True)
    
    dn.plot(kind = "barh", figsize = (12, 8), title = "features importances normalized")
    dp.plot(kind = "barh", figsize = (12, 8), title = "features importances in percentual")
    plt.show()
    
def prefitting(df, stratify = True, random_state = 42, test_size = 0.2):
    """
    Function to create the sets used for the model's estimation process.
    
    PARAMETERS:
    1) df = DataFrame Pandas object with data.
    2) stratify = Boolean parameter (default = "True") which allows to stratify the train and test set (stratification based on "y").
    3) random_state = Number to set the state of the randomic algorithm (default = 42).
    4) test_size = Proportion of total observations used to create the test set.
    
    OUTPUT:
    Four arrays with the train and test sets following this order: "X_train, X_test, y_train, y_test".
    """
    X = df.drop(columns = "target").values
    y = df["target"].values
    
    if stratify == True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state, stratify = y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    
    return X_train, X_test, y_train, y_test