# (i)PET

We have experimented with the PET and iPET approaches proposed in [Exploiting
Cloze Questions for Few Shot Text Classification and Natural Language
Inference](https://arxiv.org/abs/2001.07676) and followed their implementation
from [here](https://github.com/timoschick/pet).

## Usage

To run the scripts for this experiment, you first have to install another version of transformer. 
``` pip install transformers==3.0.2```

To train the model, the raw dataset must be pre-processed first. Execute the
script `create_dataset.py` to process data. You can only evaluate a model
trained with final classifier with the script `evaluate.py`

## Create Dataset

```sh
cd experiments/PET
python create_dataset.py\
    --dataDir DIR\    # dataset directory (raw)
    --dataType TYPE\  # reddit or regulation
    --splitSize SIZE\ # split size for labled and unlabeled data
    --outDir DIR\     # output Directory]
```

## Train a PET or iPET model

```sh
cd experiments/PET
python train.py\
    --method METHOD\                    # pet or ipet
    --pattern_ids IDS\                  # pattern ids. ex: 0 1 2
    --data_dir DIR\                     # dataset directory. provide the output directory used in the argument of create_dataset.py
    --model_type TYPE\                  # model type
    --model_name_or_path NAME_OR_PATH\  # model name or path
    --task_name NAME\                   # binary/multi
    --output_dir DIR\                   # output directory
    --do_train\                         # flag
    --do_eval\                          # flag
    --use_final_classifier              # flag. use if final classifier should be added in the pipeline
```

## Evaluate a PET or iPET model

```sh
cd experiments/PET
python evaluate.py \
    --data_dir DIR\                     # dataset directory (processd)
    --task_name NAME\                   # binary/multi
    --output_dir DIR\                   # saved output directory
    --use_final_classifier              # flag. use if final classifier should be added in the pipeline
```
