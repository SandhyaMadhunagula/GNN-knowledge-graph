import networkx as nx

# ---------- STEP 1: Normalize entity names ----------
def normalize_entity(entity):
    return entity.strip().lower().title()

# ---------- STEP 2: Sample extracted triples ----------
# (Later this will come from NER + Relation output)
triples = [
    ("Sundar Pichai", "works_for", "Google"),
    ("Google", "located_in", "California"),
    ("Tim Cook", "works_for", "Apple"),
]

# ---------- STEP 3: Build Knowledge Graph ----------
G = nx.DiGraph()

for subj, rel, obj in triples:
    subj_norm = normalize_entity(subj)
    obj_norm = normalize_entity(obj)

    # Add nodes
    G.add_node(subj_norm, label=subj)
    G.add_node(obj_norm, label=obj)

    # Add edge with relation
    G.add_edge(subj_norm, obj_norm, relation=rel)

# ---------- STEP 4: Display Graph Content ----------
print("Nodes:")
for node in G.nodes(data=True):
    print(node)

print("\nEdges:")
for edge in G.edges(data=True):
    print(edge)
