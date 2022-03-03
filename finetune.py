from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import json
from datasets import load_dataset

dataset = load_dataset("control_dataset")


nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')


# run through model pre-trained on MNLI
def tokenize(entry):
    return tokenizer(entry["premise"], entry["hypothesis"],truncation_strategy='only_first')


def metrics(evalprediction):
    print(evalprediction.predictions.shape)
    print(evalprediction.label_ids.shape)
    print(evalprediction.predictions == evalprediction.label_ids)
    return {"accuracy": (evalprediction.predictions == evalprediction.label_ids).mean()}

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