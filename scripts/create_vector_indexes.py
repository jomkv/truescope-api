import os
import libsql
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

def create_vector_indexes():
    # Force protocol to https:// to bypass sync quotas
    url = TURSO_DATABASE_URL
    if url.startswith("libsql://"):
        url = url.replace("libsql://", "https://", 1)
        
    print(f"Connecting to Turso: {url}...")
    conn = libsql.connect(url, auth_token=TURSO_AUTH_TOKEN)
    
    try:
        # Index for Claims
        print("Creating index for 'claims' table (article_chunks_idx equivalent)...")
        # Naming after the summary: 'claims_idx'
        conn.execute("""
            CREATE INDEX IF NOT EXISTS claims_idx ON claims (
                libsql_vector_idx(embedding, 'metric=cosine')
            );
        """)
        print("✅ Index creation command sent for 'claims' (claims_idx).")

        # Index for Article Chunks
        print("Creating index for 'article_chunks' table (article_chunks_idx)...")
        # Naming after the summary: 'article_chunks_idx'
        conn.execute("""
            CREATE INDEX IF NOT EXISTS article_chunks_idx ON article_chunks (
                libsql_vector_idx(embedding, 'metric=cosine')
            );
        """)
        print("✅ Index creation command sent for 'article_chunks' (article_chunks_idx).")
        
        print("\nNOTE: Index building happens in the background (DiskANN).")
        print("Wait for the 'Shadow Brain' to build before expecting search speedups.")

    except Exception as e:
        print(f"❌ Error during index creation: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    create_vector_indexes()
