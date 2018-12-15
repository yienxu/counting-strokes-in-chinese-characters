import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier

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

base_est = LogisticRegression(penalty='l1',
                              random_state=123,
                              solver='saga',
                              max_iter=10000,
                              tol=0.001,
                              multi_class='multinomial',
                              verbose=1,
                              warm_start=True,
                              C=1.0)

bagging = BaggingClassifier(base_estimator=base_est,
                            n_estimators=10,
                            max_samples=1.0,
                            max_features=1.0,
                            random_state=123,
                            n_jobs=-1,
                            bootstrap_features=False,
                            bootstrap=True,
                            verbose=1)

bagging.fit(X_train, y_train)

print()
print('test acc:')
print(bagging.score(X_test, y_test))

print('train acc:')
print(bagging.score(X_train, y_train))
