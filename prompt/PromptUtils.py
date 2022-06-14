from datasets import load_dataset
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt.data_utils import InputExample

def load_data(path):

    train = load_dataset(path, split="train[:90%]")
    dev = load_dataset(path, split="train[-10%:]")
    test = load_dataset(path, split="test")

    raw_dataset = {'train': train,
                'dev':dev,
                'test':test,
                }

    return raw_dataset

def process_reddit(raw_dataset, multiple=False):


    dataset = {}

    for split in ['train', 'test','dev']:
        dataset[split] = []
        for i, data in enumerate(raw_dataset[split]):
            
            label = data['verif']

            if label == 2 or label==3:
                continue

            if label==1:
                label = 0
            elif label==0:
                label = 1
            else:
                continue

            if data['sentence'] =='Err:509':
                continue

            if multiple:
                if label == 1:
                    if data['personal'] == 0:
                        label = 1
                    elif data['personal'] == 1:
                        label = 2

            input_example = InputExample(text_a = data['text'], label=label, guid=i)
            dataset[split].append(input_example)
        
    return dataset

def process_regulation(raw_dataset, multiple=False):

    dataset = {}

    for split in ['train', 'test','dev']:
        dataset[split] = []
        for i, data in enumerate(raw_dataset[split]):
            
            if data['label'] =='u':
                label = 0
            elif data['label'] =='e':
                label = 1
            elif multiple and data['label'] =='n':
                label = 2
            elif not multiple and data['label'] =='n':
                label = 1
            input_example = InputExample(text_a = data['text'], label=label, guid=i)
            dataset[split].append(input_example)
        
    return dataset
    
def get_verbalizer(tokenizer, multiple=False):
    
    if multiple:
        promptTemplate = ManualTemplate(
    text = '{"placeholder":"text_a"} This text contains {"mask"} information to verify the claim.' ,
    tokenizer = tokenizer,)

    else:
        promptTemplate = ManualTemplate(
        text = '{"placeholder":"text_a"} This text {"mask"} key information.' ,
        tokenizer = tokenizer)

    return promptTemplate

def get_template(tokenizer, multiple=False):

    
    if multiple:
        classes = ['u','e','n']

        promptVerbalizer = ManualVerbalizer(
            classes = classes,
            label_words = {
                "u": ["limited","insufficient",'minimal'],
                "e": ["objective","practical","factual"],
                "n": ["abstract","hypothetical",'pretend']
                
            },
            tokenizer = tokenizer,
        )

    else:
        classes = ['u','v']
        promptVerbalizer = ManualVerbalizer(
        classes = classes,
        label_words = {
            "u": ["lacks","needs","wants"],
            "v": ["contains","has","provides"],
            
        },
        tokenizer = tokenizer,)

    return classes, promptVerbalizer