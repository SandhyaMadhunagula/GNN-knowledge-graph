from datasets import load_dataset
from transformers import pipeline
from seqeval.metrics import classification_report, f1_score
import os

# -----------------------------
# 1. Load CoNLL-2003 dataset
# -----------------------------
dataset = load_dataset("conll2003")

# Use a small subset for speed (you can increase later)
test_data = dataset["test"][:100]

# -----------------------------
# 2. Load pretrained NER model
# -----------------------------
ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

# -----------------------------
# 3. Label mapping (CoNLL format)
# -----------------------------
label_names = dataset["train"].features["ner_tags"].feature.names

true_labels = []
pred_labels = []

# -----------------------------
# 4. Run NER and collect labels
# -----------------------------
for tokens, gold_tags in zip(test_data["tokens"], test_data["ner_tags"]):
    sentence = " ".join(tokens)
    predictions = ner(sentence)

    pred_seq = ["O"] * len(tokens)

    for ent in predictions:
        start = ent["start"]
        end = ent["end"]
        word = ent["word"]

        if word in tokens:
            idx = tokens.index(word)
            pred_seq[idx] = "B-" + ent["entity_group"]

    gold_seq = [label_names[tag] for tag in gold_tags]

    true_labels.append(gold_seq)
    pred_labels.append(pred_seq)

# -----------------------------
# 5. Evaluation Metrics
# -----------------------------
print("NER Evaluation on CoNLL-2003 Test Set")
print("------------------------------------")
print(classification_report(true_labels, pred_labels))
print("F1 Score:", f1_score(true_labels, pred_labels))
