import os
from typing import Optional

import torch
from datasets import load_metric
from transformers import (AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments)

from data import load_data, get_tokenizer
from model import load_model, MODEL_NAME


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)


def main(output_dir: str = "./checkpoint", sample_size: int = 1000):
    dataset = load_data(sample_size=sample_size)
    tokenizer = get_tokenizer(MODEL_NAME)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = load_model()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
    )

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = predictions.argmax(-1)
        return metric.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
