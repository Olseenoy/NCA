# src/config.py
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLUSTERING_K = 12

DB_URL = "sqlite:///./smart_nc.db"  # replace with your DB connection

RANDOM_STATE = 42
