from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B",)
                                          
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")


def map_labels(example):
    str_to_int = {'u': 0, 'n': 1, 'e': 2}
    example['label'] = str_to_int[example['label']]
    return example


def create_prompt(train, test_sentence, n_examples):
    int_to_str = {0: 'Unverifiable', 1: 'Verifiable', 2: 'Verifiable'}
    examples = random.sample(range(0, len(train)), n_examples)

    template = "Each item in the following list contains a comment and the respective verifiability. Verifiability " \
               "is one of 'verifiable' or 'unverifiable'."

    prompt = f"{template}\n"

    for i in examples:
        prompt += f"Comment: {train[i]['text']} (Verifiability: {int_to_str[train[i]['label']]})\n"
    prompt += f"Comment: {test_sentence['text']} (Verifiability: "

    return prompt


def prompt_gptj(train, test):
    n = 100
    n_examples = 3
    used_prompts = []
    results = []

    iter = 0
    while iter < n:
        prompt = create_prompt(train, test[iter], n_examples)
        max_length = len(prompt) + 32
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        gen_tokens = model.generate(input_ids,
                                    do_sample=True,
                                    temperature=0.8,
                                    max_length=max_length)
        gen_text = tokenizer.batch_decode(gen_tokens)[0]

        data = gen_text.split('\n')[n_examples + 1]

        used_prompts.append(prompt)
        results.append(data)
        iter += 1

    return used_prompts, results


def main():
    train = load_dataset("../datasets/regulation_room",
                         split="train")
    test = load_dataset("../datasets/regulation_room",
                        split="test")

    train = train.map(map_labels)
    test = test.map(map_labels)

    used_prompts, results = prompt_gptj(train, test)

    for i in results:
        print(i)

    print("\n\n\n")

    for i in used_prompts:
        print(i)

    with open('output_gpt.txt', 'w') as file:
        for prediction in results:
            file.write(prediction)
            file.write("\n")

    with open('prompts_gpt.txt', 'w') as file:
        for prompt in used_prompts:
            file.write(prompt)
            file.write("\n")


if __name__ == "__main__":
    main()
