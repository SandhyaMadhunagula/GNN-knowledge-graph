import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Path to SemEval dataset
DATA_PATH = "data/semeval_dataset/TRAIN_FILE.TXT"

sentences = []
labels = []

with open(DATA_PATH, "r", encoding="utf8") as f:
    lines = f.readlines()

for i in range(0, len(lines), 4):

    # Sentence line
    sentence_line = lines[i].strip()

    # Extract sentence
    sentence = sentence_line.split("\t")[1].strip('"')

    # Relation label
    relation = lines[i+1].strip()

    # Remove entity tags
    sentence = re.sub(r"</?e[12]>", "", sentence)

    sentences.append(sentence)
    labels.append(relation)

print("Total samples:", len(sentences))

# TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(sentences)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Train classifier
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Evaluation
pred = model.predict(X_test)

print("\nRelation Classification Report\n")
print(classification_report(y_test, pred))

# Save model
pickle.dump(model, open("relation_model.pkl", "wb"))
pickle.dump(vectorizer, open("relation_vectorizer.pkl", "wb"))

print("\nModel saved as relation_model.pkl")
print("Vectorizer saved as relation_vectorizer.pkl")