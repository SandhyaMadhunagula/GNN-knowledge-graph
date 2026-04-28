"""from transformers import pipeline
ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

text = "Sandhya lives at Hyderabad"

results = ner(text)

#for r in results:
   # print(r["word"], "->", r["entity_group"])
entities = []   # empty list to store structured entities

for r in results:
    entity = {
        "text": r["word"],
        "label": r["entity_group"]
    }
    entities.append(entity)
print(entities)"""

from transformers import pipeline
import os

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

text =  "Sundar Pichai is the CEO of Google in California."

results = ner(text)

for r in results:
    print(r["word"], "->", r["entity_group"])

