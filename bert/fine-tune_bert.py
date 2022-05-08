from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_metric

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels=3)
metric = load_metric("accuracy")


def tokenize_function(batch):
    str_to_int = {'u': 0, 'n': 1, 'e': 2}
    tokenized_batch = tokenizer(batch["text"], padding="max_length", truncation=True)
    tokenized_batch["label"] = [str_to_int[label] for label in batch["label"]]
    return tokenized_batch


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    dataset = load_dataset("../datasets/regulation_room")
    print(dataset["train"][5])
    print(dataset["train"])

    #dataset = dataset.remove_columns(["proposition_number", "comment_number", "rule_name"])
    #print(dataset["train"][5])

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    #small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(5))
    #small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(5))

    training_args = TrainingArguments(output_dir="../bert/output/test_trainer")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
        #warmup_steps=0,
        #weight_decay=0,
        #learning_rate=5e-05,
        #num_train_epochs=3,
    )

    trainer.train()


if __name__ == "__main__":
    main()
