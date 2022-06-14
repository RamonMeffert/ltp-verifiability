from datasets import load_dataset
from openprompt.plms import load_plm
from openprompt import PromptForGeneration
from openprompt.data_utils import InputExample
from openprompt import PromptDataLoader
from openprompt.prompts import PrefixTuningTemplate
from transformers import AdamW

import torch

from sklearn.metrics import classification_report

classes = ['u', 'e', 'n']

plm, tokenizer, model_config, WrapperClass = load_plm('t5', 't5-base')


promptTemplate = PrefixTuningTemplate(
    model=plm,
    text=' {"placeholder":"text_a", "shortenable": "True"} {"special": "<eos>"} {"mask"} ',
    tokenizer=tokenizer,
    using_decoder_past_key_values=False
)

promptModel = PromptForGeneration(
    template=promptTemplate,
    plm=plm,
    tokenizer=tokenizer
)

raw_dataset = load_dataset("../datasets/regulation_room")
dataset = {}
for split in ['train', 'test']:
    dataset[split] = []
    for i, data in enumerate(raw_dataset[split]):
        input_example = InputExample(text_a=data['text'],
                                     label=int(classes.index(data['label'])),
                                     guid=i)
        dataset[split].append(input_example)

wrapped_example = promptTemplate.wrap_one_example(dataset['train'][0])
print(wrapped_example)

train_data_loader = PromptDataLoader(
    dataset=dataset['train'],
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=4,
    max_seq_length=256,
    decoder_max_length=256,
    predict_eos_token=True)

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

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-4)
use_cuda = True
if use_cuda:
    promptModel = promptModel.cuda()
for epoch in range(1):
    tot_loss = 0
    for step, inputs in enumerate(train_data_loader):
        if use_cuda:
            inputs = inputs.cuda()
        loss = promptModel(inputs)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        if step % 10 == 1:
            print("Epoch {}, average loss: {}".format(epoch,
                                                      tot_loss / (step + 1)),
                  flush=True)

test_data_loader = PromptDataLoader(
    dataset=dataset['test'],
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
)

use_cuda = True
predicted = []
promptModel.eval()
with torch.no_grad():
    for batch in test_data_loader:
        if use_cuda:
            batch = batch.cuda()
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim=-1)
        #         print(preds)
        predicted.append(classes[preds])

true_labels = raw_dataset['test']['label']
print(classification_report(true_labels, predicted))
