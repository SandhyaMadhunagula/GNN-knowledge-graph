from transformers import pipeline
import pandas as pd

ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

texts = [
    "Sundar Pichai is the CEO of Google in California.",
    "Tim Cook is the CEO of Apple.",
    "Microsoft was founded by Bill Gates."
]

rows = []

for text in texts:
    results = ner(text)
    for r in results:
        rows.append({
            "sentence": text,
            "entity": r["word"],
            "entity_type": r["entity_group"]
        })

df = pd.DataFrame(rows)
df.to_csv("ner_output.csv", index=False)

print("NER output saved to ner_output.csv")
