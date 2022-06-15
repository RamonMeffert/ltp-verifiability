import re
from datasets import load_dataset
from sklearn.metrics import classification_report


def evaluate_new():
    test = load_dataset("../datasets/regulation_room", split="test[0:100]")
    #test = load_dataset("../datasets/change_my_view", split="test[0:100]")
    #re_prediction = re.compile(r"verifiable|unverifiable")
    #re_prediction = re.compile(r"non-argumentative|verifiable, personal|verifiable, non-personal|unverifiable")
    re_prediction = re.compile(r"verifiable, experiential|unverifiable|verifiable, non-experiential|experiential|non-experiential|verifiable")
    gold = []
    predictions = []
    with open("results/output_gpt_8.txt", 'r', encoding='utf8') as file:
        for line in file:
            answer = line.split("(Verifiability: ")[-1]
            match = re_prediction.search(answer.lower())
            if match:
                predictions.append(match.group().capitalize())
            else:
                predictions.append("Unverifiable")

    str_to_str = {'u': 'Unverifiable', 'n': 'Verifiable, non-experiential', 'e': 'Verifiable, experiential'}
    for i in range(100):
        label = test[i]['label']
        print(label)
        gold.append(str_to_str[label])


    for i, pred in enumerate(predictions):
        if pred == 'Experiential':
            predictions[i] = 'Verifiable, experiential'
        elif pred == 'Non-experiential' or pred == 'Verifiable':
            predictions[i] = 'Verifiable, non-experiential'

    # u - unverifiable (1, 2); v, n - verifiable, non-personal (0, 1); n - non argumentive (2,2); v, p - verifiable, personal (0, 0)
    # tpl_to_str = {(1, 2): "Unverifiable", (0, 1): "Verifiable, non-personal", (0, 0): "Verifiable, personal", (2, 2): "Non-argumentative"}
    # tpl_to_str = {(1, 2): "Unverifiable", (0, 1): "Verifiable", (0, 0): "Verifiable", (2, 2): "Unverifiable"}
    # for i in range(100):
    #     label = (test[i]['verif'], test[i]['personal'])
    #     gold.append(tpl_to_str[label])

    for i, label in enumerate(gold):
        print(predictions[i])
        print(label)
        print('\n')
    print(classification_report(gold, predictions))


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
