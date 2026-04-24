# 🏥 Clinical Trial Matcher

A semantic RAG (Retrieval-Augmented Generation) system 
that matches patients to relevant clinical trials using 
AI-powered natural language understanding.

## 🎯 Problem Statement

Finding the right clinical trial today requires manually 
reading thousands of dense medical documents. Existing 
search on ClinicalTrials.gov uses basic keyword matching 
that misses semantic connections. This system solves that 
by understanding medical meaning, not just keywords.

## ✨ What Makes It Unique

- **Not just ChatGPT** — grounded in real, live trial data
- **No hallucination** — every answer traced to actual documents  
- **Domain-specific embeddings** — BiomedBERT trained on 
  21M PubMed abstracts understands medical terminology deeply
- **Live data** — automated daily updates via GitHub Actions
- **100% free** — zero API costs, open source stack throughout

## 🏗️ Architecture
Patient Query (plain English)
→
LLM Query Validation (Llama 3.1 8B)
→
BiomedBERT Embedding (768 dimensions)
→
ChromaDB Semantic Search (cosine similarity)
→
Top 5 Matching Trials Retrieved
→
Llama 3.3 70B Reasoning (Groq)
→
Structured Match Analysis
✅ Why you qualify
⚠️  What might disqualify you
📍 Trial location
🔬 Treatment details

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Embeddings | BiomedBERT | Domain-specific medical training |
| Vector DB | ChromaDB | Local, free, persistent |
| LLM | Llama 3.3 70B | Best free reasoning model |
| Inference | Groq | Free tier, extremely fast |
| Validation | Llama 3.1 8B | Fast yes/no classification |
| Framework | LangChain | RAG orchestration |
| UI | Streamlit | Rapid ML app development |
| Updates | GitHub Actions | Free automated scheduling |
| Data | ClinicalTrials.gov API | Free US government data |

## 📊 Dataset

- **Source:** ClinicalTrials.gov (US National Library of Medicine)
- **Current size:** 1,497 trials (actively growing via daily updates)
- **Designed for:** 400k+ trials at full scale
- **Update frequency:** Daily at 2 AM UTC via GitHub Actions
- **Status filter:** Recruiting trials only

## 🚀 How To Run Locally

### Prerequisites
- Python 3.11+
- Groq API key (free at groq.com)

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/clinical-trial-matcher
cd clinical-trial-matcher

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Setup

```bash
# Create .env file
echo GROQ_API_KEY=your_key_here > .env

# Download trials data
python src/downloader.py

# Parse the data
python src/parser.py

# Embed into ChromaDB (takes 10-30 mins)
python src/embedder.py

# Run the app
streamlit run app.py
```

## 📁 Project Structure
```text
clinical-trial-matcher/
├── src/
│   ├── downloader.py   # Fetches trials from ClinicalTrials.gov
│   ├── parser.py       # Cleans and structures raw JSON
│   ├── embedder.py     # BiomedBERT embedding + ChromaDB storage
│   ├── retriever.py    # Semantic similarity search
│   ├── reasoner.py     # Groq LLM explanation generation
│   └── updater.py      # Daily delta update pipeline
├── data/
│   └── trials_parsed.json
├── embeddings/
│   └── chroma_db/      # Vector database (local)
├── .github/
│   └── workflows/
│       └── daily_update.yml
├── app.py              # Streamlit web UI
├── requirements.txt
└── README.md
```
## 🔄 Daily Update Pipeline

The system automatically stays current via GitHub Actions:

1. Runs every day at 2 AM UTC
2. Fetches only changed trials using date filter
3. Deletes stale vectors, re-embeds updated ones
4. Adds new trials, removes closed ones
5. Saves ~99.9% compute vs full daily reload

## 🔮 Future Work

- Multi-condition support (scale to 400k+ trials)
- Structured numerical eligibility validation
- Geographic radius-based trial filtering
- User profiles with saved searches
- Email alerts for new matching trials
- Chunked catchup for extended downtime scenarios

## 💡 Key Technical Decisions

**Why BiomedBERT over all-MiniLM?**
Medical terminology like HbA1c, metformin, SGLT-2 
requires domain-specific embeddings. BiomedBERT trained 
on 21M PubMed abstracts understands these terms deeply.

**Why delta updates over full reloads?**
ClinicalTrials.gov updates ~200-500 trials daily out of 
400k total. Reprocessing everything daily wastes 99.9% 
of compute. Delta updates fetch only what changed.

**Why Groq over local Ollama?**
8GB RAM laptops cannot run Llama 3 8B alongside 
ChromaDB and BiomedBERT simultaneously. Groq provides 
the same model via API at zero cost with better 
performance than local inference on consumer hardware.

## ⚠️ Disclaimer

This tool is for informational purposes only and does 
not constitute medical advice. Always consult a qualified 
healthcare professional before participating in any 
clinical trial.
