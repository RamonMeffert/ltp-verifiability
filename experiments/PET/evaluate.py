from pet.tasks import load_examples, TEST_SET, PROCESSORS, METRICS, DEFAULT_METRICS
import pet
import log
import util
from sklearn.metrics import classification_report
logger = log.get_logger('root')
import os
import numpy as np
import ast
import json
import argparse

def create_arg_parser():

    parser = argparse.ArgumentParser(description="Command line interface for PET/iPET")
    parser.add_argument("--use_final_classifier", action='store_true',
                        help="Define if final classifer is used for training")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--train_examples", default=-1, type=int,
                        help="The total number of train examples to use, where -1 equals all examples.")
    parser.add_argument("--test_examples", default=-1, type=int,
                        help="The total number of test examples to use, where -1 equals all examples.")
    parser.add_argument("--task_name", default=None, type=str, required=True, choices=PROCESSORS.keys(),
                        help="The name of the task to train/evaluate on")

    return parser.parse_args()


def enesamble_model(args, path, eval_data,reduction='wmean'):

    logits = list()
    weights = list()
    subdirs = next(os.walk(path))[1]
    for subdir in subdirs:
        model_path = os.path.join(path, subdir)
        results_file = os.path.join(path, subdir, 'results.txt')

        args.model_name_or_path = model_path
        pet_model_cfg, pet_train_cfg, pet_eval_cfg = util.load_pet_configs(args)
        wrapper = pet.init_model(pet_model_cfg)
        results = pet.evaluate(wrapper, eval_data, pet_eval_cfg)

        logits.append(results['logits'])

        if reduction == 'mean':
            result_train = 1
        else:
            with open(results_file, 'r') as fh:
                results = ast.literal_eval(fh.read())
                result_train = results['train_set_before_training']
                # print(result_train)
        weights.append(result_train)



    if reduction == 'mean':
        logits = np.mean(np.array(logits), axis=0).tolist()
    elif reduction == 'wmean':
        logits = np.average(np.array(logits), axis=0, weights=np.array(weights)).tolist()
    

    predictions = np.argmax(logits,axis=1)

    return predictions

def pred_from_logits(path):

    logitFile = os.path.join(path,'unlabeled_logits.txt')
    logits = list()
    with open(logitFile,'r') as file:
        for line in file:
            line = line.split(' ')
            if len(line) > 1:
                logit = [float(lt) for lt in line]
                logits.append(np.array(logit))


        # logits = [[float(line.split(' ')[0]), float(line.split(' ')[1])] for line in file]
    logits = np.array(logits)
    # print(logits)
    predictions = np.argmax(logits,axis=1)

    return predictions


def pred_from_logits(path, file = 'unlabeled_logits.txt'):

    logitFile = os.path.join(path,file)
    logits = list()
    with open(logitFile,'r') as file:
        for line in file:
            line = line.split(' ')
            if len(line) > 1:
                logit = [float(lt) for lt in line]
                logits.append(np.array(logit))



    logits = np.array(logits)
    # print(logits)
    predictions = np.argmax(logits,axis=1)

    return predictions

def pred_from_file(path, file = 'predictions.jsonl'):

    predFile = os.path.join(path,'final/p0-i0',file)
    with open(predFile, 'r') as file:
        predictions = [int(json.loads(jline)['label']) for jline in file]

    return predictions


def main():

    args = create_arg_parser()


    train_ex, test_ex = args.train_examples, args.test_examples
    eval_data = load_examples(args.task_name, args.data_dir, TEST_SET, num_examples=test_ex, num_examples_per_label=None)


    

    if args.use_final_classifier:

        predictions = pred_from_file(args.output_dir)
    else:

        predictions = pred_from_logits(args.output_dir)
        

    true = [int(example.label) for example in eval_data]
    

    print(classification_report(true, predictions))
    

if __name__ == "__main__":
    main()