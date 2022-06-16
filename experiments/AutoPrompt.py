from openprompt.plms import load_plm
from openprompt.prompts import ManualVerbalizer, ManualTemplate
from openprompt.prompts.prompt_generator import T5TemplateGenerator
from openprompt.pipeline_base import PromptDataLoader, PromptForClassification
import copy
import torch
from transformers import AdamW
from tqdm import tqdm
from openprompt.prompts.prompt_generator import RobertaVerbalizerGenerator
from PromptUtils import load_data, process_reddit, process_regulation
import argparse
from openprompt.prompts.prompt_generator import LMBFFTemplateGenerationTemplate


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
                        help="select the dataset type (reddit/cmv)")

    parser.add_argument(
        "-m",
        "--multiClass",
        action='store_true',
        help=
        "use this arg to format muti class dataset. by default it takes binary"
    )

    args = parser.parse_args()
    return args


def fit(model, train_dataloader, val_dataloader, loss_func, optimizer):
    best_score = 0.0
    for epoch in range(5):
        train_epoch(model, train_dataloader, loss_func, optimizer)
        score = evaluate(model, val_dataloader)
        if score > best_score:
            best_score = score
    return best_score


def train_epoch(model, train_dataloader, loss_func, optimizer, cuda=True):
    model.train()
    for step, inputs in enumerate(train_dataloader):
        if cuda:
            inputs = inputs.cuda()
        logits = model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate(model, val_dataloader, save=False, cuda=True):
    model.eval()
    allpreds = []
    alllabels = []
    with torch.no_grad():
        for step, inputs in enumerate(val_dataloader):
            if cuda:
                inputs = inputs.cuda()
            logits = model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i == j)
               for i, j in zip(allpreds, alllabels)]) / len(allpreds)

    if save:
        with open('output.txt', 'w') as file:
            for i, j in zip(allpreds, alllabels):
                file.write(f'{i},{j}\n')
    return acc


def main():

    args = create_arg_parser()

    dataDir = args.dataDir
    dataType = args.dataType
    multiClass = args.multiClass

    raw_dataset = load_data(dataDir)

    if dataType == 'reddit':
        dataset = process_reddit(raw_dataset, multiClass)

    elif dataType == 'regulation':
        dataset = process_regulation(raw_dataset, multiClass)

    # load mlm model for main tasks
    plm, tokenizer, model_config, WrapperClass = load_plm(
        "roberta", "roberta-large")

    # load generation model for template generation
    template_generate_model, template_generate_tokenizer, template_generate_model_config, template_tokenizer_wrapper = load_plm(
        't5', 't5-base')

    if not multiClass:
        num_classes = 2
        label_words = [
            [
                "lacks",
            ],
            [
                "contains",
            ],
        ]
    else:
        num_classes = 3
        label_words = [[
            "insufficient",
        ], [
            "objective",
        ], ["hypothetical"]]

    verbalizer = ManualVerbalizer(tokenizer=tokenizer,
                                  num_classes=num_classes,
                                  label_words=label_words)

    template = LMBFFTemplateGenerationTemplate(
        tokenizer=template_generate_tokenizer,
        verbalizer=verbalizer,
        text='{"placeholder":"text_a"} {"mask"} {"meta":"labelword"} {"mask"}.'
    )

    wrapped_example = template.wrap_one_example(dataset['train'][0])
    print(wrapped_example)

    cuda = True
    auto_t = True  # whether to perform automatic template generation
    auto_v = True  # whether to perform automatic label word generation

    # template generation
    if auto_t:
        print('performing auto_t...')

        if cuda:
            template_generate_model = template_generate_model.cuda()

        template_generator = T5TemplateGenerator(
            template_generate_model,
            template_generate_tokenizer,
            template_tokenizer_wrapper,
            verbalizer,
            beam_width=5
        )  # beam_width is set to 5 here for efficiency, to improve performance, try a larger number.

        dataloader = PromptDataLoader(
            dataset['train'],
            template,
            tokenizer=template_generate_tokenizer,
            tokenizer_wrapper_class=template_tokenizer_wrapper,
            batch_size=len(dataset['train']),
            decoder_max_length=128,
            max_seq_length=128,
            shuffle=False,
            teacher_forcing=False)  # register all data at once
        print('pass!')
        for data in dataloader:
            if cuda:
                data = data.cuda()
            template_generator._register_buffer(data)

        template_generate_model.eval()
        print('generating...')
        template_texts = template_generator._get_templates()

        original_template = template.text
        template_texts = [
            template_generator.convert_template(template_text,
                                                original_template)
            for template_text in template_texts
        ]
        # template_generator._show_template()
        template_generator.release_memory()
        # generate a number of candidate template text
        print(template_texts)
        # iterate over each candidate and select the best one
        best_metrics = 0.0
        best_template_text = None
        for template_text in tqdm(template_texts):
            template = ManualTemplate(tokenizer, template_text)

            train_dataloader = PromptDataLoader(
                dataset['train'],
                template,
                tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass)
            valid_dataloader = PromptDataLoader(
                dataset['validation'],
                template,
                tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass)

            model = PromptForClassification(copy.deepcopy(plm), template,
                                            verbalizer)

            loss_func = torch.nn.CrossEntropyLoss()
            no_decay = ['bias', 'LayerNorm.weight']
            # it's always good practice to set no decay to biase and LayerNorm parameters
            optimizer_grouped_parameters = [{
                'params': [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.01
            }, {
                'params': [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.0
            }]

            optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
            if cuda:
                model = model.cuda()
            score = fit(model, train_dataloader, valid_dataloader, loss_func,
                        optimizer)

            if score > best_metrics:
                print('best score:', score)
                print('template:', template_text)
                best_metrics = score
                best_template_text = template_text
        # use the best template
        template = ManualTemplate(tokenizer, text=best_template_text)
        print(best_template_text)

    # verbalizer generation

    if auto_v:
        print('performing auto_v...')
        # load generation model for template generation
        if cuda:
            plm = plm.cuda()
        verbalizer_generator = RobertaVerbalizerGenerator(
            model=plm,
            tokenizer=tokenizer,
            candidate_num=10,
            label_word_num_per_class=5)
        # to improve performace , try larger numbers

        dataloader = PromptDataLoader(dataset['train'],
                                      template,
                                      tokenizer=tokenizer,
                                      tokenizer_wrapper_class=WrapperClass,
                                      batch_size=4)
        for data in dataloader:
            if cuda:
                data = data.cuda()
            verbalizer_generator.register_buffer(data)
        label_words_list = verbalizer_generator.generate()
        verbalizer_generator.release_memory()

        # iterate over each candidate and select the best one
        current_verbalizer = copy.deepcopy(verbalizer)
        best_metrics = 0.0
        best_label_words = None
        for label_words in tqdm(label_words_list):
            current_verbalizer.label_words = label_words
            train_dataloader = PromptDataLoader(
                dataset['train'],
                template,
                tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass)
            valid_dataloader = PromptDataLoader(
                dataset['validation'],
                template,
                tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass)

            model = PromptForClassification(copy.deepcopy(plm), template,
                                            current_verbalizer)

            loss_func = torch.nn.CrossEntropyLoss()
            no_decay = ['bias', 'LayerNorm.weight']
            # it's always good practice to set no decay to biase and LayerNorm parameters
            optimizer_grouped_parameters = [{
                'params': [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.01
            }, {
                'params': [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.0
            }]

            optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
            if cuda:
                model = model.cuda()
            score = fit(model, train_dataloader, valid_dataloader, loss_func,
                        optimizer)

            if score > best_metrics:
                best_metrics = score
                best_label_words = label_words
        # use the best verbalizer
        print(best_label_words)
        verbalizer = ManualVerbalizer(tokenizer,
                                      num_classes=2,
                                      label_words=best_label_words)

    # main training loop
    train_dataloader = PromptDataLoader(dataset['train'],
                                        template,
                                        tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass)
    valid_dataloader = PromptDataLoader(dataset['validation'],
                                        template,
                                        tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass)
    test_dataloader = PromptDataLoader(dataset['test'],
                                       template,
                                       tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass)

    model = PromptForClassification(copy.deepcopy(plm), template, verbalizer)
    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        0.01
    }, {
        'params': [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        0.0
    }]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
    if cuda:
        model = model.cuda()
    score = fit(model, train_dataloader, valid_dataloader, loss_func,
                optimizer)
    test_score = evaluate(model, test_dataloader, save=True)
    print(test_score)
    
    
    
if __name__ == "__main__":
 	main()
