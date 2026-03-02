"""
preprocessing.py
----------------
Handles all data cleaning and encoding for the exam question dataset.
Run standalone: python -m src.preprocessing
"""

import os
import re
import string
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

for pkg in ("stopwords", "wordnet", "omw-1.4", "punkt"):
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg != "punkt" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

DIFFICULTY_MAP = {"Easy": 0, "Medium": 1, "Hard": 2}
DIFFICULTY_INVERSE = {v: k for k, v in DIFFICULTY_MAP.items()}
OHE_COLS = ["subject", "question_type", "cognitive_level"]
NUMERICAL_COLS = ["avg_score", "std_dev", "discrimination_index"]
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/special chars, remove stopwords, lemmatize."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)          # remove non-alpha chars
    text = re.sub(r"\s+", " ", text).strip()         # collapse whitespace
    tokens = text.split()
    tokens = [t for t in tokens if t not in _stop_words]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def build_topic_freq_map(series: pd.Series) -> dict:
    """Returns a mapping from topic value → frequency rank (descending order)."""
    freq = series.value_counts()
    return freq.to_dict()

def encode_topic(series: pd.Series, freq_map: dict) -> pd.Series:
    """Map topic → count; unseen topics → 0."""
    return series.map(freq_map).fillna(0).astype(int)

def preprocess(df: pd.DataFrame,
               is_train: bool = True,
               label_encoder: LabelEncoder = None,
               ohe_columns: list = None,
               topic_freq_map: dict = None) -> dict:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df : raw DataFrame
    is_train : if True, fit encoders; if False, use provided ones
    label_encoder : fitted LabelEncoder for 'difficulty_label' (inference)
    ohe_columns : list of column names after OHE (for alignment at inference)
    topic_freq_map : dict mapping topic → frequency (inference)

    Returns
    -------
    dict with keys:
        'df_processed'   – processed DataFrame
        'label_encoder'  – fitted LabelEncoder (or passed-through)
        'ohe_columns'    – list of OHE column names
        'topic_freq_map' – topic frequency mapping
    """
    df = df.copy()

    if "question_id" in df.columns:
        df.drop(columns=["question_id"], inplace=True)
        logger.info("Dropped 'question_id'")

    logger.info("Cleaning 'question_text' ...")
    df["cleaned_text"] = df["question_text"].apply(clean_text)
    df.drop(columns=["question_text"], inplace=True)

    if "difficulty_label" in df.columns:
        if is_train:
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.array(["Easy", "Medium", "Hard"])
            df["difficulty_encoded"] = df["difficulty_label"].map(DIFFICULTY_MAP)
        else:
            df["difficulty_encoded"] = df["difficulty_label"].map(DIFFICULTY_MAP)
        df.drop(columns=["difficulty_label"], inplace=True)
        logger.info("Encoded 'difficulty_label'")

    if is_train:
        topic_freq_map = build_topic_freq_map(df["topic"])
        logger.info(f"Built topic freq map with {len(topic_freq_map)} unique topics")

    df["topic_encoded"] = encode_topic(df["topic"], topic_freq_map)
    df.drop(columns=["topic"], inplace=True)

    logger.info(f"One-Hot Encoding: {OHE_COLS}")
    df = pd.get_dummies(df, columns=OHE_COLS, drop_first=False)

    if is_train:
        ohe_columns = df.columns.tolist()
    else:
        for col in ohe_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[[c for c in ohe_columns if c in df.columns]]

    logger.info("Preprocessing complete.")
    return {
        "df_processed": df,
        "label_encoder": label_encoder,
        "ohe_columns": ohe_columns,
        "topic_freq_map": topic_freq_map,
    }

def load_data(path: str) -> pd.DataFrame:
    """Load CSV and perform basic validation."""
    df = pd.read_csv(path)
    logger.info(f"Loaded dataset with shape {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    if "difficulty_label" in df.columns:
        logger.info(f"Class distribution:\n{df['difficulty_label'].value_counts()}")
    return df
