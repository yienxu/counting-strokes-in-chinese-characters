import os
import shutil
import sys

CHARS = None
FONT_NAME = 'simheittf.png'
PATH_TO_PNGS = '/Users/yienxu/Desktop/479data/CharactersTrimPad28/'
DEST_PATH = '/Users/yienxu/Desktop/479data/{}/'.format(FONT_NAME.split('.')[0])
print(DEST_PATH)


def loadingBar(count, total, size):
    percent = float(count) / float(total) * 100
    sys.stdout.write("\r" + str(int(count)).rjust(3, '0')
                     + "/" + str(int(total)).rjust(3, '0')
                     + ' [' + '=' * int(percent / 10) * size
                     + ' ' * (10 - int(percent / 10)) * size + ']')


if __name__ == '__main__':
    CHARS = list(filter(lambda x: not os.path.isdir(x), os.listdir(PATH_TO_PNGS)))
    total_chars = len(CHARS)
    print('We have {} characters in total.'.format(total_chars))

    if not os.path.isdir(DEST_PATH):
        os.mkdir(DEST_PATH)

    count = 0

    for i, char in enumerate(CHARS):
        if char.endswith('.DS_Store'):
            continue
        char_path = PATH_TO_PNGS + char + '/'
        font_png_path = char_path + FONT_NAME
        dst_path = DEST_PATH + char + '.png'
        if os.path.isfile(font_png_path):
            shutil.copyfile(font_png_path, dst_path)
            count += 1
        loadingBar(i, total_chars, 1)

    print("\nDone! Total PNG files copied: {}".format(count))
