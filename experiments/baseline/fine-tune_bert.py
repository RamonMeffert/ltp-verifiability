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
    #train = load_dataset("../datasets/regulation_room", split="train[:90%]")
    #eval = load_dataset("../datasets/regulation_room", split="train[-10%:]")
    #test = load_dataset("../datasets/regulation_room", split="test")
    train = load_dataset("../../datasets/regulation_room", split="train[0:3]")
    eval = load_dataset("../../datasets/regulation_room", split="train[3:4]")
    test = load_dataset("../../datasets/regulation_room", split="test[4:8]")

    tokenized_train = train.map(tokenize_function, batched=True)
    tokenized_eval = eval.map(tokenize_function, batched=True)
    tokenized_test = test.map(tokenize_function, batched=True)

    training_args = TrainingArguments(output_dir="../bert/output/test_trainer",
                                      evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
        #warmup_steps=0,
        #weight_decay=0,
        #learning_rate=5e-05,
        #num_train_epochs=3,
    )

    trainer.train()
    trainer.evaluate()

    logits, labels, _ = trainer.predict(tokenized_test)
    predictions = np.argmax(logits, axis=-1)

    print(labels)
    print(predictions)
    print(metric.compute(predictions=predictions, references=labels))


if __name__ == "__main__":
    main()
