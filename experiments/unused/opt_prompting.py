from typing import Any, Dict, List
from datasets import load_dataset
from transformers import pipeline, AutoConfig
from random import sample

# Mapping from raw label to human-text label
label_mapping = {
    0: 'verifiable',
    1: 'unverifiable',
    2: 'a non-argument',
    3: 'null'
}


def generate_list(dict_items: Dict[str, List[Any]],
                  str_format: str,
                  n: int,
                  add_label=True):
    sentences = dict_items['sentence']
    labels = dict_items['verif']

    output = []

    for i in range(n):
        if add_label:
            output.append(
                str_format.format(sentence=sentences[i],
                                  label=label_mapping[labels[i]]))
        else:
            output.append(str_format.format(sentence=sentences[i]))

    return '\n\n'.join(output)


# How many iterations to do for every prompt
n_iterations = 5

# How many examples to use
n_examples = 5

# Reddit data set
dataset = load_dataset("../datasets/change_my_view")

# Simple generator
generator = pipeline('text-generation', model="facebook/opt-1.3b", max_length=512, padding='max_length', truncation=True)

# Possible prompt templates
prompt_templates = [
"""A proposition is verifiable if it contains an objective assertion, where objective means “expressing or dealing with facts or conditions as perceived without distortion by personal feelings, prejudices, or interpretations.

Unverifiable propositions are typically opinions, suggestions, judgements, or assertions about what will happen in the future.

The following sentences are some examples of verifiable, unverifiable and non-argument propositions:

{list_examples}

{query}"""
]

# Possible formats for the list of examples
example_templates = [
    """The sentence “{sentence}” is {label}.""",
    """{label}: {sentence}""",
    """{sentence} ({label})"""
]

query_templates = [
    """The sentence “{sentence}” is """,
    """{sentence} """,
    """{sentence} """
]

# Which example template to use
example_index = 0

for template in prompt_templates:
    for n in sample(range(len(dataset['train'])), n_iterations):
        example_indices = sample(range(len(dataset['train'])), n_examples)

        list_examples = generate_list(dataset['train'][example_indices],
                                      example_templates[example_index],
                                      n_examples)

        query = generate_list(dataset['train'][[n]],
                              query_templates[example_index], 1, False)

        prompt = template.format(list_examples=list_examples, query=query)

        generated = generator(prompt)[0]['generated_text']

        new_text = generated[len(prompt):]

        print(new_text)
