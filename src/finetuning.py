from utils import get_cmu_mosei_dataset


import numpy as np


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import evaluate


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
metric = evaluate.load("accuracy")
training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    # train, test, validation = get_cmu_mosei_dataset(format_data=True)

    dataset = get_cmu_mosei_dataset(format_data=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    small_train_dataset = (
        tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    )
    small_validation_dataset = (
        tokenized_datasets["validation"].shuffle(seed=42).select(range(1000))
    )
    small_test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-cased", num_labels=7
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_validation_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
