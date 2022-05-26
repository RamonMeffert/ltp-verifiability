from datasets import load_dataset
from pprint import pprint

dataset = load_dataset("../verifiability")

pprint(dataset["train"].info)
pprint(dataset["train"][:5])
