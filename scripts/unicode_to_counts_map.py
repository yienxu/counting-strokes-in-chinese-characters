import pandas as pd

DB_PATH = '/Users/yienxu/Desktop/Dropbox/Python/STAT479/Unihan/Unihan_DictionaryLikeData.txt'
SAVE_PATH = '/Users/yienxu/Desktop/Dropbox/Python/STAT479/counts.csv'

if __name__ == '__main__':
    df = pd.read_csv(DB_PATH, encoding='UTF-8', delim_whitespace=True, comment='#',
                     names=list('1234'))

    # Keep unicode and counts only
    df = df[df['2'] == 'kTotalStrokes'].reset_index(drop=True)
    df = df.drop(['2', '4'], axis=1)
    df['3'] = pd.to_numeric(df['3'])
    df.columns = ['unicode', 'count']

    # Substitute unicode with characters
    df['unicode'] = df['unicode'].map(lambda val: chr(int(val[2:], base=16)))
    print(df)

    df.to_csv(SAVE_PATH, index=False)
