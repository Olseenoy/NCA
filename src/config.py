# src/config.py
from pathlib import Path
import os
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
try:
    load_dotenv()
except Exception:
    pass  # Prevent crash if dotenv is not available

# -----------------------------
# Directories
# -----------------------------
try:
    DATA_DIR = Path(__file__).resolve().parents[1] / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
except Exception:
    DATA_DIR = Path("./data")
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"

# -----------------------------
# Model & clustering (Safe Defaults)
# -----------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CLUSTERING_K = int(os.getenv("CLUSTERING_K", 12))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

# -----------------------------
# Database
# -----------------------------
DB_URL = os.getenv("DB_URL", "sqlite:///./smart_nc.db")

# -----------------------------
# Cloud / API authentication
# -----------------------------
GOOGLE_SHEETS_API_KEY = os.getenv("GOOGLE_SHEETS_API_KEY", "")
ONEDRIVE_CLIENT_ID = os.getenv("ONEDRIVE_CLIENT_ID", "")
ONEDRIVE_CLIENT_SECRET = os.getenv("ONEDRIVE_CLIENT_SECRET", "")
ONEDRIVE_TENANT_ID = os.getenv("ONEDRIVE_TENANT_ID", "")
ERP_API_URL = os.getenv("ERP_API_URL", "")
ERP_API_TOKEN = os.getenv("ERP_API_TOKEN", "")
