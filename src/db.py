# src/db.py
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config import DB_URL

Base = declarative_base()

class CAPA(Base):
    __tablename__ = 'capa'
    id = Column(Integer, primary_key=True, index=True)
    issue_id = Column(String, index=True)
    description = Column(Text)
    corrective_action = Column(Text)
    owner = Column(String)
    due_date = Column(DateTime)
    closed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)
