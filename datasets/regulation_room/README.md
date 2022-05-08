# Regulation Room Data Set

HuggingFace version of the data set used in the paper "Identifying Appropriate
Support for Propositions in Online User Comments" by Park and Cardie (2014).

## Usage

We first need to transform the `.txt` files to `.csv`, so HuggingFace can
process them. To do this, run `preprocess.py` from this directory.

Then, to use the data set, load it as usual:

```python
from datasets import load_dataset

data = load_dataset("datasets/regulation_room")
```

💡 _The example above assumes you are loading it from the root of this project._

## Original README

_Below is the original readme included in the data set download._

Format:
label,text#rule_name.comment_number.proposition_number

Labels:
u - unverifiable
n - non-experiential
e - experiential

For more information, please refer to "Identifying Appropriate Support for
Propositions in Online User Comments" by Joonsuk Park and Claire Cardie
(2014).

---

Park, J., & Cardie, C. (2014). Identifying Appropriate Support for Propositions
in Online User Comments. Proceedings of the First Workshop on Argumentation
Mining, 29–38. <https://doi.org/10/gg29gq>
