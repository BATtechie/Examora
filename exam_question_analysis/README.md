# Examora: Exam Question Difficulty Predictor and AI Assessment Assistant

Examora is a Streamlit-based educational intelligence app that predicts exam question difficulty (`Easy`, `Medium`, `Hard`) and provides AI-generated pedagogical feedback for better assessment design.

## Live Deployment

- Streamlit app: [https://examora.streamlit.app/](https://examora.streamlit.app/)

## What This Project Includes

- Classical ML pipeline for question difficulty prediction.
- Model comparison across Logistic Regression, Decision Tree, and Random Forest.
- Best-model artifact saving for real-time inference in the app.
- Agentic AI assistant (LangGraph + Groq + retrieval + web search) for assessment-quality guidance.
- Multi-page Streamlit UI:
  - Single Predictor
  - Batch CSV Predictor
  - Model Dashboard
  - AI Assistant

## Model Performance (Current Trained Artifacts)

Metrics below are loaded from `models/results_summary.json`.

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| Logistic Regression | 0.856 | 0.8565 |
| Decision Tree | 0.821 | 0.8212 |
| Random Forest | 0.832 | 0.8328 |

**Best model:** `Logistic Regression` (from `models/meta.json`)

## End-to-End Pipeline

1. Load dataset from `data/exam_question_dataset_5000.csv`.
2. Preprocess:
   - Drop `question_id`
   - Clean `question_text` (lowercase, remove special chars/stopwords, lemmatize)
   - Encode target (`Easy=0`, `Medium=1`, `Hard=2`)
   - Frequency encode `topic`
   - One-hot encode `subject`, `question_type`, `cognitive_level`
3. Build features:
   - TF-IDF (`max_features=300`, `ngram_range=(1,2)`)
   - Standard scaling for numerical features
   - Sparse feature matrix assembly (`scipy.sparse.hstack`)
4. Train and evaluate:
   - Logistic Regression
   - Decision Tree
   - Random Forest
5. Select best model by highest weighted F1.
6. Save artifacts in `models/` and serve them through Streamlit.

## Dataset Schema

Input dataset includes:

- `question_id`
- `subject`
- `topic`
- `question_text`
- `question_type`
- `cognitive_level`
- `avg_score`
- `std_dev`
- `discrimination_index`
- `difficulty_label` (target)

For inference (single and batch), required fields are:

- `question_text`
- `subject`
- `topic`
- `question_type`
- `cognitive_level`
- `avg_score`
- `std_dev`
- `discrimination_index`

## Streamlit App Pages

### 1) Single Predictor

- Enter one question and metadata.
- Returns predicted difficulty and model confidence (when available).

### 2) Batch Upload

- Upload CSV with required columns.
- Predicts all rows.
- Shows difficulty distribution charts.
- Export results as `examora_predictions.csv`.

### 3) Model Dashboard

- Displays per-model Accuracy and weighted F1.
- Compares model performance in charts.
- Shows confusion matrix images (generated during training).
- Highlights the best model automatically.

### 4) AI Assistant

- Uses question context plus predicted difficulty.
- Runs agentic workflow using:
  - LangGraph orchestration
  - FAISS retrieval
  - DuckDuckGo web search
  - Groq LLM (`llama-3.3-70b-versatile`)
- Returns structured output in tabs:
  - Reasoning
  - Learning Gaps
  - Recommendations

## Project Structure

```text
exam_question_analysis/
├── app/
│   ├── app.py
│   └── requirements.txt
├── assets/
│   └── architecture_diagram.png
├── data/
│   └── exam_question_dataset_5000.csv
├── models/
│   ├── best_model.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── meta.json
│   ├── results_summary.json
│   └── confusion_matrix_*.png
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── evaluate.py
│   ├── train.py
│   └── agent.py
├── requirements.txt
└── README.md
```

## Setup and Run

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Configure environment variables

Create `.env` in project root:

```env
GROQ_API_KEY="your_groq_api_key_here"
```

`GROQ_API_KEY` is required for the AI Assistant page.

### 3) Train models

```bash
python src/train.py
```

### 4) Launch Streamlit

```bash
python3 -m streamlit run app/app.py
```

## Training Output Artifacts

After training, `models/` contains:

- Best model and all candidate model `.pkl` files
- `tfidf_vectorizer.pkl`, `scaler.pkl`, `label_encoder.pkl`
- `meta.json` (best model, OHE schema, topic frequency map)
- `results_summary.json` (accuracy/F1 per model)
- Confusion matrix images for dashboard visualization

## Core Tech Stack

- Frontend/UI: `streamlit`, `plotly`
- ML/Data: `scikit-learn`, `pandas`, `numpy`, `scipy`, `nltk`, `joblib`
- Visualization during eval: `matplotlib`, `seaborn`
- AI/Agentic: `langchain`, `langgraph`, `langchain-groq`, `faiss-cpu`, `sentence-transformers`, `duckduckgo-search`
- Config: `python-dotenv`

## Deployment Notes

- Current deployed app: [https://examora.streamlit.app/](https://examora.streamlit.app/)
- For local development, run from project root using the commands above.

## Troubleshooting

- If app says model files are missing, run `python src/train.py` first.
- If AI Assistant fails, check `.env` and ensure `GROQ_API_KEY` is set.
- If NLTK resources are missing, rerun once with internet; required corpora are downloaded automatically.
