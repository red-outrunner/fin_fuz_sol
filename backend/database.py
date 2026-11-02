import os
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Get database URL from environment variable, fallback to SQLite for local development
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # Some providers (e.g. Heroku/Neon) emit 'postgres://' which SQLAlchemy 1.4+
    # doesn't accept; normalize to 'postgresql://'.
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
else:
    # Local development / tests: default to SQLite.
    DATABASE_URL = "sqlite:///./users.db"

# Branch on the actual URL scheme, not merely on whether DATABASE_URL was set — a
# SQLite URL passed via env (CI/tests) still needs check_same_thread=False, and the
# log message should reflect the real backend.
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    print("🔧 Using SQLite database")
else:
    engine = create_engine(DATABASE_URL)
    print("🚀 Using PostgreSQL database")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    """Shared typed registry for runtime queries and Alembic metadata."""

    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

