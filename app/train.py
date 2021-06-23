from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

from datasets import load_dataset, load_metric

if TYPE_CHECKING:
    from datasets.dataset_dict import DatasetDict

TASK = "ner"
BATCH_SIZE = 16
LABEL_ALL_TOKENS = True
DATASET_NAME = "wnut_17"

DATA_DIR = Path.cwd() / "data"
CHECKPOINT = DATA_DIR / "models" / "checkpoints" / "checkpoint-2500"

datasets: DatasetDict = load_dataset(DATASET_NAME)  # type: ignore

label_list = datasets["train"].features["ner_tags"].feature.names

model_checkpoint = str(CHECKPOINT)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples[f"{TASK}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the LABEL_ALL_TOKENS flag.

            else:
                label_ids.append(label[word_idx] if LABEL_ALL_TOKENS else -100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=len(label_list)
)

args = TrainingArguments(
    f"test-{TASK}",
    evaluation_strategy="epoch",  # type: ignore
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=3,
    weight_decay=0.01,
    label_names=label_list,
)

metric = load_metric("seqeval")


def compute_metrics(predictions, labels):
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=DataCollatorForTokenClassification(tokenizer),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()  # type: ignore

trainer.evaluate()  # type: ignore

predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])  # type: ignore
compute_metrics(predictions, labels)
