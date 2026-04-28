from transformers import pipeline
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load pretrained models
# -----------------------------
ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

# Simple relation logic (rule-based for demo)
def extract_relations(entities, text):
    relations = []
    persons = [e["word"] for e in entities if e["entity_group"] == "PER"]
    orgs = [e["word"] for e in entities if e["entity_group"] == "ORG"]
    locs = [e["word"] for e in entities if e["entity_group"] == "LOC"]

    for p in persons:
        for o in orgs:
            if "CEO" in text or "founder" in text:
                relations.append((p, "WORKS_AT", o))

    for o in orgs:
        for l in locs:
            relations.append((o, "LOCATED_IN", l))

    return relations

# -----------------------------
# 2. User Input
# -----------------------------
text = input("Enter text to build knowledge graph:\n")

# -----------------------------
# 3. Named Entity Recognition
# -----------------------------
entities = ner(text)

print("\nExtracted Entities:")
for e in entities:
    print(e["word"], "->", e["entity_group"])

# -----------------------------
# 4. Relation Extraction
# -----------------------------
relations = extract_relations(entities, text)

print("\nExtracted Relations:")
for r in relations:
    print(r)

# -----------------------------
# 5. Build Knowledge Graph
# -----------------------------
G = nx.DiGraph()

for e in entities:
    G.add_node(e["word"], label=e["entity_group"])

for h, r, t in relations:
    G.add_edge(h, t, label=r)

# -----------------------------
# 6. Visualization
# -----------------------------
pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(10, 7))
nx.draw(
    G, pos,
    with_labels=True,
    node_size=3000,
    node_color="lightblue",
    font_size=10,
    arrows=True
)

edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Knowledge Graph from User Text")
plt.show()
