# Experiments

## GPT-neo

To change which experiment is run, change the dataset and label set in the
source code of `template_prompting.py`. To run, execute the following:

```sh
cd experiments
python template_prompting.py
```

## Manual Prompting

```sh
cd experiments
python ManualPrompt.py\
    --dataDir DIR\   # dataset directory (raw)
    --dataType TYPE\ # reddit or regulation
    --multiClass     # flag. use to train a model for multi-label classfication
```

## Auto Prompting

```sh
cd experiments
python AutoPrompt.py\
    --dataDir DIR\   # dataset directory (raw)
    --dataType TYPE\ # reddit or regulation
    --multiClass     # flag. use to train a model for multi-label classfication
```
