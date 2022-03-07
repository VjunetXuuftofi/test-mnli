from sentence_transformers import CrossEncoder
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

def tokenize_function(examples):
    return tokenizer(examples, padding="max_length", truncation=True)

# bart-large: 0.4869
# deberta-v2-xxlarge-mnli: 0.4608
model = CrossEncoder("/data/uid1803058/models/checkpoint-42000")
correct = 0
total = 0
with open("data/test.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        prediction = model.predict([data["premise"], data["hypothesis"]])
        argmax = prediction.argmax()
        correct += data["label"] == "e" and argmax == 2 or data["label"] == "c" and argmax == 0 or data["label"] == "n" and argmax == 1
        total += 1
        print("Accuracy:", correct / total)