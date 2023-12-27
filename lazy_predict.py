import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

alg = ['HUGO', 'LSB', 'WOW', 'SUNIWARD']
i_path = [0.1, 0.2, 0.3, 0.4]
fea = [836, 1858]
tra = ['train', 'test']

for payload in i_path:
    for algo in alg:
        for nro in fea:
            for splite in tra:
                file_path_train = f'exel_output/{algo}, {payload}, {nro}, train.csv'
                df = pd.read_csv(file_path_train)
                x_train = df.iloc[:, :-1]
                y_train = df.iloc[:, -1]

                file_path_test = f'exel_output/{algo}, {payload}, {nro}, test.csv'
                df = pd.read_csv(file_path_test)
                x_test = df.iloc[:, :-1]
                y_test = df.iloc[:, -1]

                scaler = StandardScaler()
                x_train_scaled = scaler.fit_transform(x_train)
                x_test_scaled = scaler.transform(x_test)

                param_grid = {'n_neighbors': [1, 3, 5, 7, 10]}
                knn_classifier = KNeighborsClassifier()
                grid_search = GridSearchCV(knn_classifier, param_grid, cv=5)
                grid_search.fit(x_train_scaled, y_train)
                best_params = grid_search.best_params_

                final_knn_classifier = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'])
                final_knn_classifier.fit(x_train_scaled, y_train)
                y_pred = final_knn_classifier.predict(x_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"KNN {algo} {payload} {nro} Accuracy: {accuracy}")
