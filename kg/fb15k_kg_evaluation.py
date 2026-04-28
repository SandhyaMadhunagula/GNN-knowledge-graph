import random
import networkx as nx
from collections import defaultdict

# -----------------------------
# 1. Load FB15k-237 triples
# -----------------------------
# Expected format per line: head<TAB>relation<TAB>tail

file_path = "data/fb15k-237/train.txt"  # adjust path if needed

triples = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        h, r, t = line.strip().split("\t")
        triples.append((h, r, t))

print("Total triples loaded:", len(triples))

# -----------------------------
# 2. Build Knowledge Graph
# -----------------------------
G = nx.DiGraph()
for h, r, t in triples:
    G.add_edge(h, t, relation=r)

# -----------------------------
# 3. Prepare test triples
# -----------------------------
test_triples = random.sample(triples, 200)

# -----------------------------
# 4. Rule-based Link Prediction
# -----------------------------
def predict_tail(head, relation, top_k=10):
    candidates = []
    for _, neighbor, data in G.out_edges(head, data=True):
        if data["relation"] == relation:
            candidates.append(neighbor)
    return candidates[:top_k]

# -----------------------------
# 5. Compute Metrics
# -----------------------------
hits_at_10 = 0
mrr_total = 0
precision_at_10 = 0

for h, r, t in test_triples:
    predictions = predict_tail(h, r, top_k=10)

    if t in predictions:
        hits_at_10 += 1
        rank = predictions.index(t) + 1
        mrr_total += 1 / rank
        precision_at_10 += 1 / len(predictions)

num_tests = len(test_triples)

print("\nFB15k-237 KG Evaluation Results")
print("--------------------------------")
print("Hits@10:", hits_at_10 / num_tests)
print("MRR:", mrr_total / num_tests)
print("Precision@10:", precision_at_10 / num_tests)
