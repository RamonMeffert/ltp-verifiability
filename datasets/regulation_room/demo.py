from datasets import load_dataset
from pprint import pprint

dataset = load_dataset("regulation_room")

pprint(dataset["train"].info)
pprint(dataset["train"][:10])
