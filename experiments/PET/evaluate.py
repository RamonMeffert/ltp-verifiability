"""
This script can be used to train and evaluate either a regular supervised model or a PET/iPET model on
one of the supported tasks and datasets.
"""

from pet.tasks import load_examples, TEST_SET, PROCESSORS, METRICS, DEFAULT_METRICS
from pet.modeling import merge_logits_lists
import pet
import log
import util
from sklearn.metrics import classification_report
from pet.utils import save_predictions
import torch
logger = log.get_logger('root')
import os
import numpy as np
import ast

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


def main():

    args = util.create_arg_parser()
    logger.info("Parameters: {}".format(args))

    args.task_name = args.task_name.lower()
    if args.task_name not in PROCESSORS:
        raise ValueError("Task '{}' not found".format(args.task_name))
    processor = PROCESSORS[args.task_name]()
    args.label_list = processor.get_labels()

    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0

    train_ex, test_ex = args.train_examples, args.test_examples
    eval_data = load_examples(args.task_name, args.data_dir, TEST_SET, num_examples=test_ex, num_examples_per_label=None)

    args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)


    pet_model_cfg, pet_train_cfg, pet_eval_cfg = util.load_pet_configs(args)

    

    if args.use_final_classifier:
        wrapper = pet.init_model(pet_model_cfg)
        results = pet.evaluate(wrapper, eval_data, pet_eval_cfg)
        predictions = save_predictions(path=None, wrapper= wrapper, results=results)
        predictions = [pred['label'] for pred in predictions]
    else:
        # predictions = enesamble_model(args, args.model_name_or_path, eval_data)
        predictions = pred_from_logits(args.model_name_or_path)
        

    true = [int(example.label) for example in eval_data]
    

    print(classification_report(true,predictions))
    

if __name__ == "__main__":
    main()