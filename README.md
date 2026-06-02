# 🏥 MediSage AI – Hybrid Clinical Expert System
<img width="1864" height="842" alt="Screenshot 2026-05-18 051832" src="https://github.com/user-attachments/assets/c0863cdd-a9bb-4813-877a-7198a14f87cf" />

## 📌 Overview
MediSage AI is a Hybrid Neuro-Symbolic Medical Decision Support System that predicts possible diseases based on user symptoms using:

- 🧠 Rule-based reasoning (Forward & Backward Chaining)
- 📊 Symptom matching logic
- 🔍 Semantic search using Chroma Vector Database
- 🤖 LLM-powered explanation generation (Llama 3.2 via Ollama)
- 🧾 Structured medical knowledge base

This project is designed as an intelligent clinical assistant for educational and research purposes.

---

## ⚙️ Core Features

### 🧠 1. Dual Inference Engine
- Forward Chaining: Symptoms → Possible diseases
- Backward Chaining: Disease hypothesis → Symptom verification

---

### 🔍 2. Semantic Retrieval (RAG-style)
- Uses Chroma DB vector store
- Embeddings: all-MiniLM-L6-v2
- Retrieves similar medical cases

---

### 🤖 3. AI Explanation Layer
- Powered by Llama 3.2 (Ollama)
- Generates structured medical reports
- Converts raw predictions into readable clinical analysis

---

### 📊 4. Knowledge Base Engine
- Merges multiple datasets:
  - Disease-symptom mapping
  - Symptom severity
  - Disease descriptions
  - Precautions

---

### 🎨 5. Streamlit UI
- Chat-based interface
- Medical report cards
- Mode selection:
  - Forward Chaining
  - Backward Chaining

---

## 📊 Dataset Used

Kaggle Dataset:
https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset

Includes:
- Disease ↔ Symptoms mapping
- Symptom severity weights
- Disease descriptions
- Medical precautions

---

## 🧠 System Architecture

User Symptoms
→ Streamlit UI (app.py)
→ Inference Engine (ma.py)
→ Forward / Backward Chaining Logic
→ Chroma Vector Database
→ Knowledge Base (CSV + JSON)
→ LLM (Llama 3.2 via Ollama)
→ Final Medical Report

---

## 📁 Project Structure

```

MediSage-AI/
│
├── app.py
├── ma.py
├── pre.py
├── requirements.txt
│
├── DB/
├── chroma_medical_db/
│
├── expert_knowledge_base.csv
├── cleaned_severity.csv
├── severity_dict.json
│
└── README.md

````

---

## 🚀 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
````

### Build knowledge base + vector DB

```bash
python pre.py
```

### Run Streamlit app

```bash
streamlit run app.py
```

---

## 🧪 Example Input

```
fever, headache, stomach pain
```

## 📌 Output

* Predicted diseases
* Confidence scores
* Matched symptoms
* Missing symptoms
* Medical precautions
* AI-generated explanation

---

## 🧠 Technologies Used

* Python
* Streamlit
* Pandas
* LangChain
* Chroma DB
* HuggingFace Embeddings
* Ollama (Llama 3.2)

---

## ⚠️ Disclaimer

This project is for educational and research purposes only.
It is NOT a medical diagnosis tool.
