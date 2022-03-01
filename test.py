from sentence_transformers import CrossEncoder
import json
# bart-large: 0.4869
model = CrossEncoder("microsoft/deberta-xxlarge-v2-mnli")
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