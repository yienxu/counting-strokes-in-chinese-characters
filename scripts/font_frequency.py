import argparse
import heapq
import os
import pickle
import sys

import numpy as np
import pandas as pd

CHARS = None
HEAP = []
FREQ = {}
PATH = None


def build_parser():
    p = argparse.ArgumentParser(
        description='Get the number of characters that a font supports.')
    p.add_argument('path_to_dataset',
                   metavar='PATH',
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

    PATH = args.path_to_dataset[0] + '/'
    CHARS = list(filter(lambda x: not os.path.isdir(x), os.listdir(PATH)))

    total_chars = len(CHARS)
    print('We have {} characters in total.'.format(total_chars))

    if args.generate:
        for i, char in enumerate(CHARS):
            char_path = PATH + char + '/'
            fonts = os.listdir(char_path)
            for font in fonts:
                add_char_to_font(char, font)
            loadingBar(i, total_chars, 1)
        with open('freq.pickle', 'wb') as f:
            pickle.dump(obj=FREQ, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    elif args.view:
        with open('freq.pickle', 'rb') as f:
            FREQ = pickle.load(f)

    for k, v in FREQ.items():
        heapq.heappush(HEAP, (-len(v), k))

    for i in range(args.view):
        count, font = heapq.heappop(HEAP)
        count = -count
        print(count, font)
