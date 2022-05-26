import re
from datasets import load_dataset
from sklearn.metrics import classification_report


def main():
    test = load_dataset("../datasets/regulation_room", split="test[0:100]")
    re_prediction = re.compile(r"Verifiable|Unverifiable")
    gold = []
    predictions = []
    with open("results/output_gpt_2.txt", 'r', encoding='utf8') as file:
        for line in file:
            answer = line.split("(Verifiability:")[1]
            match = re_prediction.search(answer)
            if match:
                predictions.append(match.group())
            else:
                predictions.append("Unverifiable")

    str_to_str = {'u': 'Unverifiable', 'n': 'Verifiable', 'e': 'Verifiable'}
    for i in range(100):
        label = test[i]['label']
        gold.append(str_to_str[label])

    print(classification_report(gold, predictions))



if __name__ == "__main__":
    main()
