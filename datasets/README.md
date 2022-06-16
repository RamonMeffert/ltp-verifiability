# Datasets

## [Regulation Room](regulation_room/)

The [original dataset files][rr-dataset] used by Park & Cardie (2014) are
available from Joonsuk Park's [website][rr-site]. The code used to transform the
provided `.txt` files into `.csv` files that can be used with HuggingFace
datasets is available in [`preprocess.py`][rr-pp] and the dataset file is
available in [`regulation_room.py`][rr-hf]. Exploratory analysis is available in
[`analysis.ipynb`][rr-nb].

## [Change My View](change_my_view/)

This is an as yet unpublished dataset created at the University of Groningen.
The original data files were already in `.csv` format, and the dataset file is
available in [`change_my_view.py`][cmv-hf]. Exploratory analysis is available in
[`analysis.ipynb`][cmv-nb].

## [Merged dataset](verifiability/)

To combat the small size of the datasets, we have created a merged dataset from
the two datasets mentioned above. The code used to generate the `.csv` files for
the splits is available in [`generate.py`][merged-pp], and the dataset file is
available in [`verifiability.py`][merged-hf]. Exploratory analysis is available
in [`analysis.ipynb`][merged-nb].

_Note: this dataset is not used in the paper._

<!-- URLS -->

[rr-site]: https://facultystaff.richmond.edu/~jpark/
[rr-dataset]: https://facultystaff.richmond.edu/~jpark/data/jpark_aclw14.zip
[rr-hf]: datasets/regulation_room/regulation_room.py
[rr-pp]: datasets/regulation_room/preprocess.py
[rr-nb]: datasets/regulation_room/analysis.ipynb
[cmv-hf]: datasets/change_my_view/change_my_view.py
[cmv-nb]: datasets/change_my_view/analysis.ipynb
[merged-hf]: datasets/verifiability/verifiability.py
[merged-pp]: datasets/verifiability/generate.py
[merged-nb]: datasets/verifiability/analysis.ipynb
