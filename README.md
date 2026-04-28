

# GNN Knowledge Graph
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Model](https://img.shields.io/badge/Model-REBEL%20%7C%20BART-orange?logo=huggingface)
![GNN](https://img.shields.io/badge/Graph%20Logic-NetworkX%20%7C%20R--GCN-red)
![Framework](https://img.shields.io/badge/Framework-Flask-lightgrey?logo=flask)
![Visualization](https://img.shields.io/badge/Visualization-PyVis-blueviolet)

This project provides an end-to-end framework for transforming unstructured text documents (PDF, DOCX, TXT) into interactive, structured Knowledge Graphs.  

By leveraging state-of-the-art Transformer models and Graph Neural Networks, the system identifies complex relationships between entities and visualizes them in a dynamic web interface.

---

## 🚀 Features

- **Joint Entity & Relation Extraction**  
  Powered by the REBEL (BART-based) model to extract Subject-Relation-Object (SPO) triplets in a single step.

- **Graph Management**  
  Implemented with NetworkX for efficient graph structure handling.

- **Graph Reasoning**  
  Uses Relational Graph Convolutional Networks (R-GCN) for semantic analysis.

- **Interactive UI**  
  Flask-based web interface with dynamic graph visualization using PyVis.

---

## 🛠️ Tech Stack

- **Languages:** Python  
- **AI/ML:** PyTorch, HuggingFace Transformers, NetworkX, R-GCN  
- **Backend:** Flask, Pickle  
- **Frontend:** HTML, CSS, PyVis  

---

## 🚀 Getting Started

### Step 1: Clone the Repository
```bash
git clone https://github.com/SandhyaMadhunagula/gnn-kg.git
cd gnn-kg
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
python app.py
```

## 📸 Project Output

### 🖥️ User Interface
<p align="center">
  <img src="https://github.com/user-attachments/assets/6f790f53-8af3-43d0-8bd2-36ea35b14436" width="800"/>
</p>

> Interactive Knowledge Graph Dashboard featuring physics-based node clustering.

---

### 🔍 Semantic Querying
<p align="center">
  <img src="https://github.com/user-attachments/assets/6ab1c6ed-ee6a-435b-b0bd-f6d25a54e6fb" width="400"/>
</p>

> AI-powered Query Engine demonstrating semantic relationship discovery.

---

### 🗄️ Extracted Triplets (Database View)
<p align="center">
  <img src="https://github.com/user-attachments/assets/22d12d2a-ef50-4f19-9abf-c3eb132b1fce" width="450"/>
</p>

> Structured Triplet View: Detailed breakdown of the extracted Knowledge Base.

---

## 🏗️ Architecture 
<p align="center">
  <img src="https://github.com/user-attachments/assets/478071d5-9f3d-4d83-aac6-a33ced778405" width="450"/>
</p>

### 1️⃣ Data Ingestion Layer
Processes raw documents (PDF, DOCX, TXT) through a cleaning and tokenization pipeline. Ensures text is optimized for transformer input constraints.

### 2️⃣ Extraction Layer (REBEL & BART)
Utilizes BART Encoder-Decoder architecture to perform **joint extraction** of Subject-Relation-Object triplets in a single step, reducing cascading errors.

### 3️⃣ Knowledge Management
Extracted triplets are structured into a graph using NetworkX and stored efficiently using Pickle serialization.

### 4️⃣ Reasoning Layer (R-GCN)
Applies Relational Graph Convolutional Networks for semantic reasoning, enabling link prediction and deeper graph insights.

### 5️⃣ User Interface
Flask-based web interface with PyVis visualization for interactive graph exploration and relationship discovery.

---

## 🧠 How It Works

1. 📄 Input documents (PDF, DOCX, TXT) are uploaded through the web interface  
2. 🤖 NLP model (REBEL) extracts Subject-Relation-Object triplets  
3. 🕸️ Triplets are converted into a graph structure using NetworkX  
4. 🧠 R-GCN processes the graph for deeper semantic understanding  
5. 📊 The graph is visualized interactively using PyVis in the browser


## 🎯 Use Cases

- **Research Synthesis:** Automating knowledge extraction from large volumes of academic papers  
- **Intelligent Search:** Powering semantic query systems for multi-hop relationship discovery  
- **Automated Document Understanding:** Converting unstructured corporate files into structured data  
- **AI Knowledge Bases:** Building the underlying "brain" for intelligent agents and chatbots  
