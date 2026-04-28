import re
import pickle
import torch
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import pickle
import torch
from transformers import BertTokenizer, BertModel
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

DATA_PATH = "data/semeval_dataset/TRAIN_FILE.TXT"

sentences = []
labels = []

with open(DATA_PATH, "r", encoding="utf8") as f:
    lines = f.readlines()

for i in range(0, len(lines), 4):

    sentence_line = lines[i].strip()
    sentence = sentence_line.split("\t")[1].strip('"')
    relation = lines[i+1].strip()

    sentence = re.sub(r"</?e[12]>", "", sentence)

    sentences.append(sentence)
    labels.append(relation)

print("Total samples:", len(sentences))

# Load BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

embeddings = []

print("Generating BERT embeddings...")

for sentence in sentences:

    inputs = tokenizer(sentence,
                       return_tensors="pt",
                       truncation=True,
                       padding=True,
                       max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[:,0,:].squeeze().numpy()
    embeddings.append(cls_embedding)

#X = embeddings
X = np.array(embeddings)
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = LogisticRegression(max_iter=2000)

clf.fit(X_train, y_train)


pred = clf.predict(X_test)

print("\nRelation Classification Report\n")
print(classification_report(y_test, pred))

pickle.dump(clf, open("relation_model_bert.pkl","wb"))
pickle.dump(tokenizer, open("bert_tokenizer.pkl","wb"))

print("\nSaved relation_model_bert.pkl")
DATA_PATH = "data/semeval_dataset/TRAIN_FILE.TXT"

sentences = []
labels = []

with open(DATA_PATH, "r", encoding="utf8") as f:
    lines = f.readlines()

for i in range(0, len(lines), 4):

    sentence_line = lines[i].strip()
    sentence = sentence_line.split("\t")[1].strip('"')
    relation = lines[i+1].strip()

    sentence = re.sub(r"</?e[12]>", "", sentence)

    sentences.append(sentence)
    labels.append(relation)

print("Total samples:", len(sentences))

# Load BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

embeddings = []

print("Generating BERT embeddings...")

for sentence in sentences:

    inputs = tokenizer(sentence,
                       return_tensors="pt",
                       truncation=True,
                       padding=True,
                       max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[:,0,:].squeeze().numpy()
    embeddings.append(cls_embedding)

X = embeddings
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = LogisticRegression(max_iter=2000)

clf.fit(X_train, y_train)


pred = clf.predict(X_test)

print("\nRelation Classification Report\n")
print(classification_report(y_test, pred))

pickle.dump(clf, open("relation_model_bert.pkl","wb"))
pickle.dump(tokenizer, open("bert_tokenizer.pkl","wb"))

print("\nSaved relation_model_bert.pkl")