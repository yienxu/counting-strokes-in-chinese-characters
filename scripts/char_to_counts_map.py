import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt

COUNT_DATA_PATH = '/Users/yienxu/Desktop/Dropbox/Python/STAT479/counts.csv'
PNG_DIR = '/Users/yienxu/Desktop/Dropbox/Python/STAT479/simheittf/'
SAVE_PATH = '/Users/yienxu/Desktop/Dropbox/Python/STAT479/dataset.csv'

if __name__ == '__main__':
    df = pd.read_csv(COUNT_DATA_PATH, encoding='UTF-8')
    print(df.head(5))

    unicodes = []
    counts = []
    img_arrays = []

    for idx, row in df.iterrows():
        char = row['unicode']
        filename = "".join([PNG_DIR, char, '.png'])
        if not os.path.isfile(filename):
            continue
        img = plt.imread(filename)
        img_array = img.flatten()
        img_arrays.append(img_array)
        counts.append(row['count'])
        unicodes.append(char)

    img_arrays = np.array(img_arrays)
    print(img_arrays.shape)

    new_arrays = []
    for i in range(img_arrays.shape[1]):
        arr = img_arrays[:, i].tolist()
        new_arrays.append(arr)

    img_arrays = new_arrays

    data = [unicodes, counts, *img_arrays]
    d = {}
    for idx, col in enumerate(data):
        d[idx] = col
    dataset = pd.DataFrame(d)
    dataset.columns = ['char', 'count', *list(range(1, 28 ** 2 + 1))]

    print(dataset)
    dataset.to_csv(SAVE_PATH, index=False, encoding='UTF-8')
