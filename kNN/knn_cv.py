import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

df = pd.read_csv('data/dataset.csv', encoding='utf-8')

print('numdata = {}\n'.format(np.sum((df['count'] > 30) | (df['count'] == 1))))

df = df[(df['count'] <= 30) & (df['count'] != 1)]
print(df.shape)

X = df.iloc[:, 2:28 ** 2 + 2].astype(np.float)
y = df['count'].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    random_state=123,
                                                    shuffle=True,
                                                    stratify=y)

pipe = make_pipeline(PCA(),
                     KNeighborsClassifier())

# First run
# param_grid = {'pca__n_components': [100, 200, 300, 400, 500, 600, 700, 784],
#               'kneighborsclassifier__n_neighbors': np.arange(5, 601, step=50)}

# Second run
param_grid = {'pca__n_components': [300, 400, 500],
              'kneighborsclassifier__n_neighbors': np.arange(5, 60, step=5)}

gsearch = GridSearchCV(pipe,
                       param_grid=param_grid,
                       refit=True,
                       iid=False,
                       cv=5,
                       n_jobs=-1,
                       verbose=1)

gsearch.fit(X_train, y_train)

print('best params:')
print(gsearch.best_params_)

print()
print('test acc:')
print(gsearch.score(X_test, y_test))

print('train acc:')
print(gsearch.score(X_train, y_train))
