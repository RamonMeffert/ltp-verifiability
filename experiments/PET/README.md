We have experimented the PET and iPET approaces proposed in [Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference](https://arxiv.org/abs/2001.07676)
and follwed their implemention from [here](https://github.com/timoschick/pet).

# Usage

 To train the model, the raw dataset must be pre-processed first. Execute the script ```create_dataset.py``` to process data. 
 You can only evaluate a model trained with final classifier with the script `evaluate.py`


## Create Dataset

```
cd experiment/PET

python3 create_dataset.py \
--dataDir [dataset directory (raw)]\
--dataType [reddit or regulation]\
--splitSize [split size for labled and unlabeled data]
--outDir [output Directory]
```

## Train a PET or iPET model

```
cd experiment/PET

python3 train.py \
--method [pet or ipet] \
--pattern_ids [pattern ids. ex: 0 1 2] \
--data_dir [dataset directory] \ # provide the output directory used in the argument of create_dataset.py
--model_type [model type] \
--model_name_or_path [model name or path] \
--task_name [binary/multi] \
--output_dir [output directory] \
--do_train \
--do_eval \
--use_final_classifier [use if final classifier should be added in the pipeline]

```
## Evaluate a PET or iPET model


```
cd experiment/PET
python3 evaluate.py \
--method [pet or ipet] \
--data_dir [dataset directory (processd)] \
--model_type [model type] \
--model_name_or_path [saved model name or path] \
--task_name [binary/multi] \
--output_dir [saved output directory] \

```
