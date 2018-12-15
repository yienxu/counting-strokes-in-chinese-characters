## Scripts related to our dataset

This directory contains scripts to generate the dataset that we use to train our model. There are four scripts in total and each will be explained in detail as follows.

#### The original dataset

The [original dataset](https://blog.usejournal.com/making-of-a-chinese-characters-dataset-92d4065cc7cc) contains 15 million 28x28 PNG files of 52,835 Chinese characters with different fonts.

#### `font_frequency.py`

This script is used to visualize how many `.png` files there are associated with each font in the original dataset.

#### `font_pngs.py`

This script is used to extract every character image of a specific font from the original dataset.

#### `unicode_to_counts_map.py`

This script is used to generate a data frame whose rows contain information about the number of strokes a character has. Stroke counts data is retrieved from the [Unihan Database](https://www.unicode.org/charts/unihan.html).

#### `gen_dataset.py`

This script is used to generate our final dataset, which contains a character, its stroke counts, and 784 (28*28) columns, each corresponding to a pixel in the image.

#### `df_visualization.py`

This script is used to visualize our dataset. It plots a histogram of character counts categorized by stroke numbers.
