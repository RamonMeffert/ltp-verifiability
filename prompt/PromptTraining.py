from datasets import load_dataset
from pprint import pprint
from openprompt.plms import load_plm
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt.data_utils import InputExample
from openprompt import PromptDataLoader

import torch
from transformers import  AdamW, get_linear_schedule_with_warmup


from sklearn.metrics import classification_report

classes = ['u','e','n']


plm, tokenizer, model_config, WrapperClass = load_plm('bert',"distilbert-base-cased")

from openprompt.prompts import ManualTemplate
promptTemplate = ManualTemplate(
    text = '{"placeholder":"text_a"} The comment is {"mask"}',
    tokenizer = tokenizer,
)


promptVerbalizer = ManualVerbalizer(
    classes = classes,
    label_words = {
        "u": ["unverifiable","unprovable","unsupportable"],
        "e": ["experiential","factual"],
        "n": ["non-experiential", "hypothetical"],
    },
    tokenizer = tokenizer,
)


promptModel = PromptForClassification(
    template = promptTemplate,
    plm = plm,
    verbalizer = promptVerbalizer,
)



raw_dataset = load_dataset("ltp-verifiability/datasets/regulation_room")
dataset = {}
for split in ['train', 'test']:
    dataset[split] = []
    for i, data in enumerate(raw_dataset[split]):
        input_example = InputExample(text_a = data['text'], label=int(classes.index(data['label'])), guid=i)
        dataset[split].append(input_example)
        
print(dataset['train'][0])



train_data_loader = PromptDataLoader(
        dataset = dataset['train'],
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size = 4
#         max_seq_length=256, decoder_max_length=3,
#         batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False,
#         truncate_method="head"
    
    )


loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in promptModel.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in promptModel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]


optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
use_cuda = False
if use_cuda:
    promptModel =  promptModel.cuda()
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
        if step %1000 ==1:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)


test_data_loader = PromptDataLoader(
        dataset = dataset['test'],
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
#         max_seq_length=256, decoder_max_length=3,
#         batch_size=1,shuffle=False, teacher_forcing=False, predict_eos_token=False,
#         truncate_method="head"
    
    )


use_cuda = False
predicted = []
promptModel.eval()
with torch.no_grad():
    for batch in test_data_loader:
        if use_cuda:
            batch = batch.cuda()
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim = -1)
#         print(preds)
        predicted.append(classes[preds])



true_labels = raw_dataset['test']['label']
print(classification_report(true_labels, predicted))




