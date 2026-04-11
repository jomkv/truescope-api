import os
import libsql
from dotenv import load_dotenv

load_dotenv()

def get_schema():
    url = os.getenv("TURSO_DATABASE_URL")
    token = os.getenv("TURSO_AUTH_TOKEN")
    conn = libsql.connect(url, auth_token=token)
    
    try:
        # Check claims schema
        res = conn.execute("SELECT sql FROM sqlite_master WHERE name='claims'")
        print(f"Claims SQL: {res.fetchone()}")
        
        # Check article_chunks schema
        res = conn.execute("SELECT sql FROM sqlite_master WHERE name='article_chunks'")
        print(f"ArticleChunks SQL: {res.fetchone()}")
        
        # Check column info
        res = conn.execute("PRAGMA table_info(claims)")
        print(f"Claims PRAGMA: {res.fetchall()}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    get_schema()
