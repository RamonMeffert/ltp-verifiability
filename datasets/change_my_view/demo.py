from datasets import load_dataset
from pprint import pprint

dataset = load_dataset("datasets/change_my_view")

pprint(dataset["train"].info)
pprint(dataset["train"][:10])
