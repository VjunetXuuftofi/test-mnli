from sentence_transformers import CrossEncoder
model = CrossEncoder("facebook/bart-large-mnli")
print(model.predict(["He likes red.", "She likes red."]))