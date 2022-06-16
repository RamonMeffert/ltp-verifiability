import os
import torch
from pet.tasks import PROCESSORS, load_examples, UNLABELED_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS, FULL_TRAIN_SET
from pet.utils import eq_div
import pet
import log
import util

logger = log.get_logger('root')


def main():

    args = util.create_arg_parser()
    logger.info("Parameters: {}".format(args))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in PROCESSORS:
        raise ValueError("Task '{}' not found".format(args.task_name))
    processor = PROCESSORS[args.task_name]()
    args.label_list = processor.get_labels()

    train_ex_per_label, test_ex_per_label = None, None
    train_ex, test_ex = args.train_examples, args.test_examples
    if args.split_examples_evenly:
        train_ex_per_label = eq_div(args.train_examples, len(args.label_list)) if args.train_examples != -1 else -1
        test_ex_per_label = eq_div(args.test_examples, len(args.label_list)) if args.test_examples != -1 else -1
        train_ex, test_ex = None, None

    eval_set = TEST_SET if args.eval_set == 'test' else DEV_SET


    if not args.use_final_classifier:

        train_data = load_examples(args.task_name, args.data_dir, FULL_TRAIN_SET, num_examples=train_ex, num_examples_per_label=train_ex_per_label)
        eval_data = load_examples(args.task_name, args.data_dir, DEV_SET, num_examples=test_ex, num_examples_per_label=test_ex_per_label)
        unlabeled_data = load_examples(args.task_name, args.data_dir, TEST_SET, num_examples=args.unlabeled_examples)

    else:
        train_data = load_examples(args.task_name, args.data_dir, TRAIN_SET, num_examples=train_ex, num_examples_per_label=train_ex_per_label)
        eval_data = load_examples(args.task_name, args.data_dir, TEST_SET, num_examples=test_ex, num_examples_per_label=test_ex_per_label)
        unlabeled_data = load_examples(args.task_name, args.data_dir, UNLABELED_SET, num_examples=args.unlabeled_examples)

    args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)

    pet_model_cfg, pet_train_cfg, pet_eval_cfg = util.load_pet_configs(args)
    sc_model_cfg, sc_train_cfg, sc_eval_cfg = util.load_sequence_classifier_configs(args)
    ipet_cfg = util.load_ipet_config(args)

    if args.method == 'pet':
        pet.train_pet(pet_model_cfg, pet_train_cfg, pet_eval_cfg, sc_model_cfg, sc_train_cfg, sc_eval_cfg,
                      pattern_ids=args.pattern_ids, output_dir=args.output_dir,
                      ensemble_repetitions=args.pet_repetitions, final_repetitions=args.sc_repetitions,
                      reduction=args.reduction, train_data=train_data, unlabeled_data=unlabeled_data,
                      eval_data=eval_data, do_train=args.do_train, do_eval=args.do_eval,
                      no_distillation=args.no_distillation, seed=args.seed)

    elif args.method == 'ipet':
        pet.train_ipet(pet_model_cfg, pet_train_cfg, pet_eval_cfg, ipet_cfg, sc_model_cfg, sc_train_cfg, sc_eval_cfg,
                       pattern_ids=args.pattern_ids, output_dir=args.output_dir,
                       ensemble_repetitions=args.pet_repetitions, final_repetitions=args.sc_repetitions,
                       reduction=args.reduction, train_data=train_data, unlabeled_data=unlabeled_data,
                       eval_data=eval_data, do_train=args.do_train, do_eval=args.do_eval, seed=args.seed)

    else:
        raise ValueError(f"Training method '{args.method}' not implemented")


if __name__ == "__main__":
    main()


