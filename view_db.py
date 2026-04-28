import pickle
import pandas as pd

with open("graph_database.pkl", "rb") as f:
    G = pickle.load(f)

# Convert edges to a list of dictionaries
edge_list = [{"Subject": s, "Relation": d['label'], "Object": t} for s, t, d in G.edges(data=True)]

# Display as a table
df = pd.DataFrame(edge_list)
print(df)