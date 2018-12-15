import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('data/dataset.csv', encoding='utf-8')


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
    #     plt.savefig('counts.eps')
    plt.show()


print('numdata = {}\n'.format(np.sum((df['count'] > 30) | (df['count'] == 1))))

df = df[(df['count'] <= 30) & (df['count'] != 1)]
plot_df_counts(df)
print(df.shape)
