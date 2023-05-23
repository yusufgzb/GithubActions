import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def knn_experiment(n_neighbors, weights, algorithm, metric):
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, metric=metric)
    
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy score: {accuracy}")
n_neighbors_list = [3, 5, 10]
weights_list = ['uniform', 'distance']
algorithm_list = ['auto', 'ball_tree','brute']
metric_list = ['euclidean', 'manhattan']
results = []

for n_neighbors in n_neighbors_list:
    for weights in weights_list:
        for algorithm in algorithm_list:
            for metric in metric_list:
                knn_experiment(n_neighbors, weights, algorithm, metric)
                
