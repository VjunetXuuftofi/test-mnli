
model = CrossEncoder("facebook/bart-large-mnli")
model.predict(["He likes red.", "She likes red."])