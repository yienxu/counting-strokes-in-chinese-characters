import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

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

est = LogisticRegressionCV(Cs=5,
                           cv=5,
                           penalty='l1',
                           solver='saga',
                           max_iter=500,
                           tol=0.05,
                           n_jobs=-1,
                           verbose=1,
                           refit=True,
                           multi_class='multinomial',
                           random_state=123)

est.fit(X_train, y_train)

print()
print('test acc:')
print(est.score(X_test, y_test))

print('train acc:')
print(est.score(X_train, y_train))
