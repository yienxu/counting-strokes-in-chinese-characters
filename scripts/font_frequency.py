import argparse
import os
import pickle
import sys

import pandas as pd

FREQ = {}


def build_parser():
    p = argparse.ArgumentParser(
        description='Get the number of characters that a font supports.')
    p.add_argument('path_to_dataset',
                   metavar='PATH_TO_PNGS',
                   type=str,
                   nargs=1,
                   help='path to the super large dataset')
    p.add_argument('-g', '--generate',
                   action='store_true',
                   help='generate the freq dictionary')
    p.add_argument('-v', '--view',
                   action='store',
                   type=int,
                   help='view the freq dictionary')
    return p


def add_char_to_font(char, font):
    if FREQ.get(font) is None:
        FREQ[font] = [char]
    else:
        FREQ[font].append(char)


def loadingBar(count, total, size):
    percent = float(count) / float(total) * 100
    sys.stdout.write("\r" + str(int(count)).rjust(3, '0')
                     + "/" + str(int(total)).rjust(3, '0')
                     + ' [' + '=' * int(percent / 10) * size
                     + ' ' * (10 - int(percent / 10)) * size + ']')


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    path = args.path_to_dataset[0] + '/'
    chars = list(filter(lambda x: not os.path.isdir(x), os.listdir(path)))

    total_chars = len(chars)
    print('We have {} characters in total.'.format(total_chars))

    if args.generate:
        for i, char in enumerate(chars):
            if char.endswith('.DS_Store'):
                continue
            char_path = path + char + '/'
            fonts = os.listdir(char_path)
            for font in fonts:
                add_char_to_font(char, font)
            loadingBar(i, total_chars, 1)
        with open('freq.pickle', 'wb') as f:
            pickle.dump(obj=FREQ, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    elif args.view:
        with open('freq.pickle', 'rb') as f:
            FREQ = pickle.load(f)

    counts = []
    fonts = []

    for k, v in FREQ.items():
        counts.append(len(v))
        fonts.append(k)

    df = pd.DataFrame({'Count': counts, 'Font': fonts})
    df = df.sort_values(by=['Count'], ascending=False)
    if args.generate:
        df.to_csv('freq.csv', index=False)
    else:
        print(df.head(args.view))
