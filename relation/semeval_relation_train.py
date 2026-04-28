import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from datasets import load_dataset

# ---------------------------------------
# 1. Load SemEval-2010 Task 8 dataset
# ---------------------------------------
dataset = load_dataset("sem_eval_2010_task_8")

# Use a subset for speed
data = dataset["train"][:2000]

sentences = data["sentence"]
labels = data["relation"]

# ---------------------------------------
# 2. Train-Test Split
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    sentences, labels, test_size=0.2, random_state=42
)

# ---------------------------------------
# 3. TF-IDF Vectorization
# ---------------------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------------------
# 4. Train Logistic Regression Classifier
# ---------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ---------------------------------------
# 5. Evaluation
# ---------------------------------------
y_pred = model.predict(X_test_vec)

print("Relation Classification Evaluation (SemEval-2010)")
print("------------------------------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
