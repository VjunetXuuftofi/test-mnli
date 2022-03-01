from sentence_transformers import CrossEncoder
model = CrossEncoder("facebook/bart-large-mnli")
print(model.predict(["He likes red.", "She likes red."]))
print(model.predict(["He likes red.", "He likes red."]))
print(model.predict(["He likes red.", "He doesn't like red."]))