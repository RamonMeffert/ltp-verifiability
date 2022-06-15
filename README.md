# A Comparison of Prompt Engineering Methods for Checking Verifiability of Online User Comments

Md Zobaer Hossain, Linfei Zhang, Robert van Timmeren and Ramon Meffert, June 2022

This repository contains the source code for the experiments, data processing
and data analysis conducted as part of our course project for the 2021–2022
edition of the _Language Technology Project_ course at the University of
Groningen.

## Data

All files related to the datasets are located in the [datasets](datasets/)
folder. We have taken the original dataset files and transformed them into the
[HuggingFace dataset format][hf-datasets]. All dataset folders contain the
original dataset files, an analysis notebook and a demo file showing how you use
the dataset.

### [Regulation Room](datasets/regulation_room/)

The [original dataset files][rr-dataset] used by Park & Cardie (2014) are
available from Joonsuk Park's [website][rr-site]. The code used to transform the
provided `.txt` files into `.csv` files that can be used with HuggingFace
datasets is available in [`preprocess.py`][rr-pp] and the dataset file is
available in [`regulation_room.py`][rr-hf]. Exploratory analysis is available in
[`analysis.ipynb`][rr-nb].

### [Change My View](datasets/change_my_view/)

This is an as yet unpublished dataset created at the University of Groningen.
The original data files were already in `.csv` format, and the dataset file is
available in [`change_my_view.py`][cmv-hf]. Exploratory analysis is available in
[`analysis.ipynb`][cmv-nb].

### [Merged dataset](datasets/verifiability/)

To combat the small size of the datasets, we have created a merged dataset from
the two datasets mentioned above. The code used to generate the `.csv` files for
the splits is available in [`generate.py`][merged-pp], and the dataset file is
available in [`verifiability.py`][merged-hf]. Exploratory analysis is available
in [`analysis.ipynb`][merged-nb].

_Note: this dataset is not used in the paper._

## Experiments

All code for experiments is located in the [experiments](experiments/) folder.

- Baseline: BERT
- GPT-neo
- (i)PET
- LM-BFF
- Prompt Training

---

## References

Park, J., & Cardie, C. (2014). Identifying Appropriate Support for Propositions
in Online User Comments. _Proceedings of the First Workshop on Argumentation
Mining_, 29–38. <https://doi.org/10/gg29gq>

<!-- URLs -->

[hf-datasets]: https://huggingface.co/docs/datasets/dataset_script
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
