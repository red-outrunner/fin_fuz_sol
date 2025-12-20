import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get database URL from environment variable, fallback to SQLite for local development
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # Production: Use Neon PostgreSQL (or any PostgreSQL database)
    # Neon provides URLs in format: postgresql://user:pass@host/db
    # Some providers use 'postgres://' which SQLAlchemy 1.4+ doesn't support
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    engine = create_engine(DATABASE_URL)
    print("🚀 Using PostgreSQL database (production)")
else:
    # Local development: Use SQLite
    DATABASE_URL = "sqlite:///./users.db"
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False}  # Only needed for SQLite
    )
    print("🔧 Using SQLite database (local development)")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

