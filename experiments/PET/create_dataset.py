from datasets import load_dataset
import pandas as pd
import argparse
import os

def create_arg_parser():

    """Returns a map with commandline parameters taken from the user"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataDir", required=True, type=str, help="provide the path of dataset directory"
    )
    parser.add_argument(
        "-t", "--dataType", default='reddit', choices=['reddit','regulation'] ,type=str, help="select the dataset type (reddit/cmv)"
    )

    parser.add_argument(
        "-o", "--outDir", required=True,type=str, help="provide the desired output dataset directory"
    )

    parser.add_argument(
        "-m", "--multiClass", action='store_true', help="use this arg to format muti class dataset. by default it takes binary"
    )

    parser.add_argument(
        "-s", "--splitSize", default=1000, type=int ,help="split size for labled and unlabeled data"
    )


    args = parser.parse_args()
    return args



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

            input_example = [label,data['sentence'].replace('\n',' ')]
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
            input_example = [label,data['text']]
            dataset[split].append(input_example)
        
    return dataset
    

def save_data(outDir, dataset, split=1000):
     
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    test = pd.DataFrame(dataset['test'],columns=['label','text_a'])
    test.to_csv(f'{outDir}/test.csv', index=False, header=False) 

    dev = pd.DataFrame(dataset['dev'],columns=['label','text_a'])
    dev.to_csv(f'{outDir}/dev.csv', index=False, header=False) 

    train = pd.DataFrame(dataset['train'],columns=['label','text_a'])
    train.to_csv(f'{outDir}/full-train.csv',index=False, header=False) 

    unlabled = dataset['train'][split:]
    train = dataset['train'][:split]

            
    train = pd.DataFrame(train,columns=['label','text_a'])
    train.to_csv(f'{outDir}/train.csv',index=False, header=False) 

    unlabled = pd.DataFrame(unlabled,columns=['label','text_a'])
    unlabled.to_csv(f'{outDir}/unlabeled.csv',index=False, header=False) 


def main():

    args = create_arg_parser()

    dataDir = args.dataDir
    outDir = args.outDir
    dataType = args.dataType
    multiClass = args.multiClass
    split = args.splitSize

    raw_dataset = load_data(dataDir)

    if dataType=='reddit':
        dataset = process_reddit(raw_dataset, multiClass)
        save_data(outDir, dataset, split)
    elif dataType =='regulation':
        dataset = process_regulation(raw_dataset, multiClass)
        save_data(outDir, dataset, split)

    



if __name__ == "__main__":

    main()