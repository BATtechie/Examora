# 🎓 Examora — Exam Question Difficulty Predictor & Assessment Design Assistant

Examora is a full-stack educational tool featuring a classical Machine Learning pipeline to predict the difficulty level (**Easy / Medium / Hard**) of exam questions, alongside an **Agentic AI Assistant** that autonomously reasons about assessment quality, retrieves pedagogical best practices, and suggests constructive improvements.

Built with Scikit-Learn, LangGraph, Groq API (LLaMa 3), NLTK, Streamlit, and Plotly.

---

## 🌟 Key Features

### 1. 🤖 Agentic AI Assessment Assistant
- **Conversational Workflow:** Uses a LangGraph state machine to process assessment queries organically.
- **Pedagogical RAG:** Leverages FAISS vector stores and HuggingFace embeddings to retrieve scientifically backed teaching strategies.
- **Real-Time Web Search:** Integrates DuckDuckGo to pull the latest educational context for specific topics.
- **Actionable Insights:** Uses Groq's `llama-3.3-70b-versatile` to generate beautifully formatted, concise feedback separated into: *Reasoning*, *Learning Gaps*, and *Recommendations*.

### 2. 🔮 ML Difficulty Predictor
- Predicts exam question difficulty based on NLP analysis of the question text and historical student metrics.
- Uses TF-IDF vectorization and Standard Scaling.
- Trains and compares Logistic Regression, Decision Tree, and Random Forest classifiers.

### 3. 📊 Interactive Streamlit Hub
- Multi-page application with a dynamic dark glassmorphism UI.
- **Single Predictor:** Input raw question metadata for instant inference.
- **Batch CSV:** Upload bulk spreadsheets for automated grading, complete with Plotly charts and downloadable CSV results.
- **Dashboard:** Dive into Model metrics (F1, Accuracy, Confusion Matrices).

---

## 🏗️ Project Architecture

![Examora System Architecture](assets/architecture_diagram.png)

**System Overview (Detailed):**

```text
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                             │
│  CSV Dataset (5000 Qs)          │          User Web Form Input   │
└──────────┬──────────────────────┼──────────────────────┬────────┘
           │                      │                      │
           ├──────────────────────┼──────────────────────┤
           │        PREPROCESSING PIPELINE               │
           │  • Text Cleaning (lowercase, lemmatize)     │
           │  • Label Encoding (Easy→0, Medium→1, Hard→2)│
           │  • Categorical Encoding (Frequency/OHE)    │
           └──────────────────────┬──────────────────────┘
                                  │
           ┌──────────────────────┴──────────────────────┐
           │      FEATURE ENGINEERING                   │
           │  • TF-IDF Vectorization (300 features)     │
           │  • Standard Scaling (numerical)            │
           │  • Sparse Matrix Assembly (scipy.hstack)   │
           └──────────────────────┬──────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │           MODEL TRAINING                          │
        │  • Logistic Regression, Decision Tree, Forest     │
        │  • Best Model Selection by F1 Score               │
        └──────────────────────┬──────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
      ┌─────────────────┐             ┌─────────────────┐
      │  INFERENCE DB   │             │   AGENTIC AI    │
      │  (Predictions)  │             │  (LangGraph)    │
      └─────────────────┘             └─────────────────┘
               │                               │
               ▼                               ▼
      ┌─────────────────────────────────────────────────┐
      │               STREAMLIT WEB UI                  │
      │   (Single, Batch, Dashboard, AI Assistant)      │
      └─────────────────────────────────────────────────┘
```

---

## 🗂 Project Structure

```text
exam_question_analysis/
│
├── app/
│   └── app.py                  # Streamlit web application (UI)
│
├── src/
│   ├── __init__.py             
│   ├── preprocessing.py        # Data cleaning, encoding, text preprocessing
│   ├── feature_engineering.py  # TF-IDF, scaling, feature matrix assembly
│   ├── evaluate.py             # Metrics & confusion matrix visualizer 
│   ├── train.py                # Full training pipeline
│   └── agent.py                # LangGraph Agentic AI State Machine
│
├── data/
│   └── exam_question_dataset_5000.csv   
│
├── models/                     # Auto-generated ML artifacts
│   ├── best_model.pkl          
│   ├── tfidf_vectorizer.pkl    
│   ├── scaler.pkl              
│   └── meta.json               
│
├── .env.example                # Example environment variables (Groq Key)
├── requirements.txt            # All Python dependencies
└── README.md                   # This file
```

---

## ⚙️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory to enable the Agentic AI:
```env
GROQ_API_KEY="your_groq_api_key_here"
```

### 3. Train the Classical ML Models (Optional if pre-trained)
```bash
python src/train.py
```

### 4. Launch the Web App
```bash
python3 -m streamlit run app/app.py
```
*You can also access the live hosted version if deployed on Streamlit Community Cloud!*

---

## 🛠️ Technologies & Libraries

| Library | Purpose |
|---|---|
| `scikit-learn` | ML models, TF-IDF, scaler, metrics, label encoder |
| `pandas` & `numpy` | Data loading, manipulation, matrix ops |
| `nltk` | Stopword removal, lemmatization |
| `streamlit` | Multi-page web UI framework |
| `plotly` | Interactive data charts |
| `langchain` + `langgraph` | Agent workflow state management and execution |
| `langchain-groq` | API integration for LLaMa 3 reasoning |
| `duckduckgo-search` | Live pedagogical web context retrieval |
| `faiss-cpu` | Vector Similarity Search (RAG) |
| `sentence-transformers` | HuggingFace local model embeddings |
| `python-dotenv` | Secure environment variable handling |

---

## 🔑 Key Design Decisions

1. **Explicit AI State Management (LangGraph):** Moving beyond basic LLM chains, Examora uses a directed acyclic graph to pass state iteratively between context-retrieval (FAISS), web-research (DuckDuckGo), and final generation nodes to massively decrease hallucinations.
2. **Frequency Encoding for Topic:** Given the high cardinality of the topic column, frequency encoding avoids the dimensionality explosion that would occur with standard One-Hot Encoding.
3. **Structured Agentic Tabs:** Groq is explicitly prompted to natively output clean, parsable JSON matching three UI constraints (Reasoning, Learning Gaps, Recommendations) for seamless integration into Streamlit Tabs.
4. **Gaussian Pipeline Noise:** Random noise is injected into synthetic dataset numeric columns at training time to prevent pure memorization, resulting in realistic ~85% accuracy floors. 
5. **Class Imbalance Mitigation:** Standard classification balancing arguments (`class_weight="balanced"`) combined with stratified test splits ensure fairness across the Easy/Medium/Hard target space.
