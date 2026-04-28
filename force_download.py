from flask import Flask, render_template, request
import networkx as nx
from pyvis.network import Network
import os, re, pickle
from PyPDF2 import PdfReader
from docx import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# 1. Load the "Brain" (REBEL-large)
model_name = "Babelscape/rebel-large" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# --- DATABASE PERSISTENCE LOGIC ---
DB_FILE = "graph_database.pkl"

def save_db(graph):
    with open(DB_FILE, "wb") as f:
        pickle.dump(graph, f)

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return nx.DiGraph()

# Initialize graph from "Database"
G = load_db()
# ----------------------------------

def extract_triples(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    gen_kwargs = {"max_length": 256, "length_penalty": 0, "num_beams": 3, "num_return_sequences": 1}
    generated_tokens = model.generate(**inputs, **gen_kwargs)
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)[0]

    triples = []
    current_subject, current_relation, current_object = "", "", ""
    current_state = ""

    for token in decoded.replace("<s>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current_state = "s"
            if current_subject and current_relation and current_object:
                triples.append((current_subject.strip(), current_relation.strip(), current_object.strip()))
            current_subject, current_relation, current_object = "", "", ""
        elif token == "<subj>": current_state = "o"
        elif token == "<obj>": current_state = "r"
        else:
            if current_state == "s": current_subject += " " + token
            elif current_state == "o": current_object += " " + token
            elif current_state == "r": current_relation += " " + token

    if current_subject and current_relation and current_object:
        triples.append((current_subject.strip(), current_relation.strip(), current_object.strip()))
    return triples

def visualize_graph():
    net = Network(height="600px", width="100%", directed=True, bgcolor="#ffffff", font_color="black")
    for node in G.nodes():
        net.add_node(node, label=node, color="#97C2FC")
    for source, target, data in G.edges(data=True):
        net.add_edge(source, target, label=data.get("label", ""))
    
    if not os.path.exists("static"): os.makedirs("static")
    net.save_graph("static/graph.html")

@app.route("/", methods=["GET", "POST"])
def index():
    global G
    answer = None
    
    if request.method == "POST":
        text = ""
        # Handle inputs
        if "text" in request.form and request.form["text"].strip():
            text = request.form["text"]
        elif "file" in request.files:
            file = request.files["file"]
            if file.filename.lower().endswith(".pdf"):
                text = " ".join([p.extract_text() for p in PdfReader(file).pages if p.extract_text()])
            elif file.filename.lower().endswith(".docx"):
                text = " ".join([p.text for p in Document(file).paragraphs])

        if text:
            # We don't G.clear() anymore because we want to keep adding to the database!
            sentences = re.split(r'[.!?]', text)
            for sent in sentences:
                if len(sent.strip()) > 10:
                    for s, r, o in extract_triples(sent):
                        G.add_edge(s, o, label=r)
            save_db(G) # Save after every update
            visualize_graph()

        if "query" in request.form:
            user_query = request.form["query"].lower()
            for node in G.nodes():
                if node.lower() in user_query:
                    facts = [f"{G[node][nb]['label']} {nb}" for nb in G.successors(node)]
                    if facts:
                        answer = f"According to the database, {node}: " + ", ".join(facts)
    
    # Always visualize existing graph on load
    if len(G.nodes) > 0: visualize_graph()
    
    return render_template("index.html", answer=answer, graph=os.path.exists("static/graph.html"))

if __name__ == "__main__":
    app.run(debug=True)