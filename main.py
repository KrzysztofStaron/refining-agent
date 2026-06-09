from tribev2 import TribeModel

# Downloads weights from HuggingFace (facebook/tribev2) on first run
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

# Build events from the text file, then predict
df = model.get_events_dataframe(text_path="my_text.txt")
preds, segments = model.predict(events=df)

print(preds.shape)  # (n_timesteps, n_vertices) — ~20k vertices on fsaverage5