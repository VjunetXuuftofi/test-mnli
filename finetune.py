from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import json
from datasets import load_dataset
import numpy as np

dataset = load_dataset("control_dataset")


nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')


# run through model pre-trained on MNLI
def tokenize(entry):
    return tokenizer(entry["premise"], entry["hypothesis"],truncation_strategy='only_first')


def metrics(evalprediction):
    correct = 0.
    print(evalprediction.predictions)
    print(evalprediction.predictions[0].shape)
    print(evalprediction.label_ids)
    print(evalprediction.label_ids[0].shape)
    for i in range(len(evalprediction.predictions)):
        correct += np.sum(np.argmax(evalprediction.predictions[i], axis=0) == evalprediction.label_ids[i])
    return {"accuracy": correct/len(evalprediction.predictions)}

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_data = dataset.map(tokenize)

training_args = TrainingArguments(
    output_dir="/data/uid1804058/models/",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy='steps',
    eval_steps=50
)

trainer = Trainer(
    model=nli_model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    tokenizer=tokenizer,
    compute_metrics=metrics,
    data_collator=data_collator,
)

trainer.train()

trainer.evaluate()