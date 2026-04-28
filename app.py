from flask import Flask, render_template, request, redirect, url_for, session
import networkx as nx
from pyvis.network import Network
import os, re, pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import csv
from flask import Response
import io

app = Flask(__name__)
app.secret_key = "secret_key_for_session"

model_name = "Babelscape/rebel-large" 
device = "cuda" if torch.cuda.is_available() else "cpu"



load_dotenv() # This loads the variables from .env
HF_TOKEN = os.getenv("HF_TOKEN")
rebel_tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

#rebel_tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
rebel_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HF_TOKEN, low_cpu_mem_usage=True).to(device)


DB_FILE = "graph_database.pkl"

def save_db(graph):
    with open(DB_FILE, "wb") as f:
        pickle.dump(graph, f)

def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "rb") as f:
                return pickle.load(f)
        except: return nx.DiGraph()
    return nx.DiGraph()

G = load_db()

def extract_triples(text):
    inputs = rebel_tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    gen_kwargs = {"max_length": 128, "length_penalty": 0, "num_beams": 1, "num_return_sequences": 1}
    generated_tokens = rebel_model.generate(**inputs, **gen_kwargs)
    decoded = rebel_tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)[0]

    triples = []
    current_subject, current_relation, current_object = "", "", ""
    current_state = ""
    
    # ADD THESE TWO LINES TO FIX THE "FIRST WORD" PROBLEM
    clean_decoded = decoded.replace("<s>", "").replace("</s>", "")
    clean_decoded = clean_decoded.replace("<triplet>", " <triplet> ").replace("<subj>", " <subj> ").replace("<obj>", " <obj> ")

    # CHANGE THIS LOOP TO USE clean_decoded
    for token in clean_decoded.split():
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
    net = Network(height="600px", width="100%", directed=True, bgcolor="#ffffff", font_color="black", cdn_resources='remote')
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=150, damping=0.4)
    
    # CRITICAL FIX: Loop through nodes and edges to draw them
    for node in G.nodes():
        net.add_node(node, label=node, color="#00d2ff", size=25, shadow={'enabled': True, 'color': 'rgba(0,210,255,0.6)', 'size': 10})
    for source, target, data in G.edges(data=True):
        net.add_edge(source, target, label=data.get("label", ""), color="#a29bfe")
    
    if not os.path.exists("static"): os.makedirs("static")
    net.save_graph("static/graph.html")

@app.route("/", methods=["GET", "POST"])
def index():
    global G
    answer = None
    user_query = ""
    text = session.get('user_text', "") 

    if request.method == "POST":
        # 1. HANDLE FILE UPLOAD OR TEXT BOX
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            ext = file.filename.split('.')[-1].lower()
            if ext == "pdf":
                reader = PdfReader(file)
                text = " ".join([page.extract_text() for page in reader.pages])
            elif ext == "docx":
                text = " ".join([p.text for p in Document(file).paragraphs])
            elif ext == "txt":
                text = file.read().decode("utf-8")
        elif "text" in request.form and request.form["text"].strip():
            text = request.form["text"]
       
        # 2. PROCESS DATA (Only if we have new text)
        if text and "query" not in request.form:
            session['user_text'] = text
            sentences = [s.strip() for s in re.split(r'[\n.!?]', text) if len(s.strip()) > 10]
            print(f"--- 🚀 AI is extracting from {len(sentences)} sentences ---")
            for i, sent in enumerate(sentences):
                print(f"📄 Processing {i+1}/{len(sentences)}...")
                for s, r, o in extract_triples(sent):
                    G.add_edge(s.title().strip(), o.title().strip(), label=r.strip())
            save_db(G)
            visualize_graph()

        # 3. HANDLE SEARCH QUERY
        if "query" in request.form:
            user_query = request.form["query"].strip()
            keywords = [w.lower() for w in user_query.split() if len(w) > 3]
            results = []
            for node in G.nodes():
                if any(k in node.lower() for k in keywords):
                    for n in G.successors(node):
                        results.append(f"<b>{node}</b> {G[node][n]['label']} <b>{n}</b>")
                    for p in G.predecessors(node):
                        results.append(f"<b>{p}</b> {G[p][node]['label']} <b>{node}</b>")
            answer = " • " + "<br> • ".join(list(set(results))[:8]) if results else f"Nothing found for '{user_query}'."

    db_triples = [{"s": s, "r": d['label'], "o": t} for s, t, d in G.edges(data=True)]
    return render_template("index.html", answer=answer, graph=os.path.exists("static/graph.html"), user_query=user_query, user_text=text, db_triples=db_triples)

@app.route("/export_csv")
def export_csv():
    # 1. Create a string buffer to hold CSV data
    output = io.StringIO()
    writer = csv.writer(output)
    
    # 2. Write the Header
    writer.writerow(['Subject', 'Relationship', 'Object'])
    
    # 3. Write the Data from the Graph G
    for s, t, d in G.edges(data=True):
        writer.writerow([s, d.get('label', ''), t])
    
    # 4. Prepare the response for download
    output.seek(0)
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=knowledge_graph.csv"}
    )
@app.route("/clear")
def clear_db():
    global G
    G = nx.DiGraph()
    session.clear()
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    if os.path.exists("static/graph.html"): os.remove("static/graph.html")
    return redirect(url_for('index'))

if __name__ == "__main__":
    # 0.0.0.0 makes it accessible to the internet
    app.run(host="0.0.0.0", port=7860, debug=True)
