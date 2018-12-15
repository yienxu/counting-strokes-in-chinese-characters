import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline

from skimage.measure import block_reduce

df = pd.read_csv('data/dataset.csv', encoding='utf-8')

df.head()

print('numdata = {}\n'.format(np.sum((df['count'] > 30) | (df['count'] == 1))))
df = df[(df['count'] <= 30) & (df['count'] != 1)]
print(df.shape)


def plot_df_counts(df):
    x_ticks = np.asarray(list(set(df['count'])))
    xx = np.arange(np.max(x_ticks) + 1)
    yy = np.bincount(df['count'])

    for x, y in zip(xx, yy):
        print("{}->{}\t".format(x, y), end='')

    plt.bar(xx, yy)
    plt.title('Stroke Counts of Characters')
    plt.xlabel('Number of Strokes')
    plt.ylabel('Number of Characters')
    plt.show()


# plot_df_counts(df)


X = df.iloc[:, 2:28 ** 2 + 2].astype(np.float)
y = df['count'].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    random_state=123,
                                                    shuffle=True,
                                                    stratify=y)


class MaxPooler(BaseEstimator):

    def __init__(self, kernel=None, vis=False):
        self.kernel = kernel
        self.DIM = (28, 28)
        self.vis = vis

    def fit_transform(self, X, y=None):
        return self.transform(X=X, y=y)

    def transform(self, X, y=None):
        imgs = []
        for i, img_df in X.iterrows():
            img = img_df.values.reshape(self.DIM)
            img_mp = block_reduce(img, self.kernel, np.max)
            imgs.append(img_mp.flatten())
            if self.vis:
                print(img_mp.shape)
                plt.imshow(img_mp, cmap='gray')
                vis = False
        new_df = pd.DataFrame(data=imgs)
        new_df.columns = list(range(1, imgs[0].size + 1))
        return new_df

    def fit(self, X, y=None):
        return self


pipe = make_pipeline(MaxPooler(),
                     LogisticRegression(penalty='l1',
                                        random_state=123,
                                        solver='saga',
                                        max_iter=500,
                                        tol=0.05,
                                        multi_class='multinomial',
                                        verbose=1,
                                        warm_start=True,
                                        n_jobs=-1))

param_grid = {'maxpooler__kernel': [(2, 2), (3, 3), (4, 4)],
              'logisticregression__C': np.logspace(-4, 4, num=5)}

gsearch = GridSearchCV(pipe,
                       param_grid=param_grid,
                       refit=True,
                       iid=False,
                       cv=5,
                       n_jobs=1,
                       verbose=1)

gsearch.fit(X_train, y_train)

print('best params:')
print(gsearch.best_params_)

print()
print('test acc:')
print(gsearch.score(X_test, y_test))

print('train acc:')
print(gsearch.score(X_train, y_train))
