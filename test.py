from sentence_transformers import CrossEncoder
import json
model = CrossEncoder("facebook/bart-large-mnli")
correct = 0
total = 0
with open("data/test.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        prediction = model.predict([data["sentence1"], data["sentence2"]])
        argmax = prediction.argmax()
        correct += data["label"] == "e" and argmax == 2 or data["label"] == "c" and argmax == 0 or data["label"] == "n" and argmax == 1
        total += 1
        print("Accuracy:", correct / total)