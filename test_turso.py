import os
import json
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

def test_connection():
    db_url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")
    
    if not db_url or not auth_token:
        print("❌ Missing TURSO_DATABASE_URL or TURSO_AUTH_TOKEN in .env")
        return

    # Clean the host
    host = db_url.replace("libsql://", "").replace("https://", "").replace("http://", "").strip("/")
    
    # Try the standard sqlalchemy-libsql construction
    uri = f"sqlite+libsql://{host}?authToken={auth_token}"
    print(f"Testing URI: sqlite+libsql://{host}?authToken=***")
    
    try:
        engine = create_engine(uri)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).fetchone()
            print(f"✅ Success! Connection test result: {result}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        
    print("-" * 30)
    print("Trying alternative with explicit https protocol...")
    uri_https = f"sqlite+libsql://https://{host}?authToken={auth_token}"
    try:
        engine = create_engine(uri_https)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).fetchone()
            print(f"✅ Success with HTTPS! Connection test result: {result}")
    except Exception as e:
        print(f"❌ HTTPS failed: {e}")

if __name__ == "__main__":
    test_connection()
