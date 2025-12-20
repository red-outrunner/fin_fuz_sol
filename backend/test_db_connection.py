import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Add the backend directory to sys.path to import database.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import DATABASE_URL, engine

def test_connection():
    print(f"Checking connection to: {DATABASE_URL[:20]}...")
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT version();"))
            version = result.fetchone()
            print("✅ Connection Successful!")
            print(f"📊 PostgreSQL Version: {version[0]}")
            return True
    except Exception as e:
        print("❌ Connection Failed")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    load_dotenv()
    test_connection()
