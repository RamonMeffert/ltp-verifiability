from openprompt.plms import load_plm
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
import torch
from transformers import AdamW
from sklearn.metrics import classification_report
from PromptUtils import get_template, get_verbalizer, process_reddit, process_regulation, load_data
import argparse


def create_arg_parser():
    """Returns a map with commandline parameters taken from the user"""

    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--dataDir",
                        required=True,
                        type=str,
                        help="provide the path of dataset directory")
    parser.add_argument("-t",
                        "--dataType",
                        default='reddit',
                        choices=['reddit', 'regulation'],
                        type=str,
                        help="select the dataset type (reddit/regulation)")

    parser.add_argument(
        "-m",
        "--multiClass",
        action='store_true',
        help=
        "use this arg to format muti class dataset. by default it takes binary"
    )

    args = parser.parse_args()
    return args


def train(promptModel, train_data_loader, use_cuda=True):

    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in promptModel.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        0.01
    }, {
        'params': [
            p for n, p in promptModel.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        0.0
    }]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)

    if use_cuda:
        promptModel = promptModel.cuda()
    for epoch in range(5):
        tot_loss = 0
        for step, inputs in enumerate(train_data_loader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = promptModel(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step % 1000 == 1:
                print("Epoch {}, average loss: {}".format(
                    epoch, tot_loss / (step + 1)),
                      flush=True)


def eval(promptModel, test_data_loader, use_cuda=True):

    use_cuda = True
    predicted = []
    promptModel.eval()
    with torch.no_grad():
        for batch in test_data_loader:
            if use_cuda:
                batch = batch.cuda()
            logits = promptModel(batch)
            preds = torch.argmax(logits, dim=-1)
            predicted.append(preds)

    return predicted


def main():

    plm, tokenizer, model_config, WrapperClass = load_plm(
        'roberta', "roberta-base")

    args = create_arg_parser()

    dataDir = args.dataDir
    dataType = args.dataType
    multiClass = args.multiClass

    promptTemplate = get_template(tokenizer, multiClass)

    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=get_verbalizer(tokenizer, multiClass)[1],
    )

    raw_dataset = load_data(dataDir)

    if dataType == 'reddit':
        dataset = process_reddit(raw_dataset, multiClass)

    elif dataType == 'regulation':
        dataset = process_regulation(raw_dataset, multiClass)

    train_data_loader = PromptDataLoader(dataset=dataset['train'],
                                         tokenizer=tokenizer,
                                         template=promptTemplate,
                                         tokenizer_wrapper_class=WrapperClass,
                                         batch_size=4)

    test_data_loader = PromptDataLoader(
        dataset=dataset['test'],
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
    )

    train(promptModel, train_data_loader)
    predicted = eval(promptModel, test_data_loader)

    true_labels = dataset['test']['label']
    print(classification_report(true_labels, predicted))
    
 


if __name__ == "__main__":
 	main()
