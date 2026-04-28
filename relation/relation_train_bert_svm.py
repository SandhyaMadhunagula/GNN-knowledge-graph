import re
import pickle
import torch
import numpy as np

from transformers import BertTokenizer, BertModel
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

# ===============================
# Dataset Path
# ===============================
DATA_PATH = "data/semeval_dataset/TRAIN_FILE.TXT"

sentences = []
labels = []

# ===============================
# Load SemEval Dataset
# ===============================
with open(DATA_PATH, "r", encoding="utf8") as f:
    lines = f.readlines()

for i in range(0, len(lines), 4):

    sentence_line = lines[i].strip()

    # Extract sentence text
    sentence = sentence_line.split("\t")[1].strip('"')

    # Extract relation label
    relation = lines[i+1].strip()

    # Remove entity tags
    sentence = re.sub(r"</?e[12]>", "", sentence)

    sentences.append(sentence)
    labels.append(relation)

print("Total samples:", len(sentences))


# ===============================
# Load BERT Model
# ===============================
print("Loading BERT model...")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

model.eval()


# ===============================
# Generate BERT Embeddings
# ===============================
print("Generating BERT embeddings...")

embeddings = []

for sentence in sentences:

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # CLS token embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    embeddings.append(cls_embedding)

# Convert to numpy array
X = np.array(embeddings)
y = np.array(labels)


# ===============================
# Train Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ===============================
# Train SVM Classifier
# ===============================
print("Training SVM classifier...")

#clf = SVC(kernel="rbf", probability=True)
clf = LinearSVC()

clf.fit(X_train, y_train)


# ===============================
# Evaluation
# ===============================
pred = clf.predict(X_test)

print("\nRelation Classification Report\n")

print(classification_report(y_test, pred, zero_division=0))


# ===============================
# Save Model
# ===============================
pickle.dump(clf, open("relation_model_svm.pkl", "wb"))

print("\nSaved model: relation_model_svm.pkl")