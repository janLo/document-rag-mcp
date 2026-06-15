import sqlite3
import hashlib

def main():
    conn = sqlite3.connect("data/state.db")
    query = "Personalampel"
    cursor = conn.execute("SELECT chunk_id, bm25(chunks_fts) as score FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score ASC", (query,))
    rows = cursor.fetchall()
    for row in rows:
        chunk_id = row[0]
        score = row[1]
        
        # Get length from files
        c = conn.execute("SELECT length(text) FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        length = c[0] if c else 0
        
        # Get filename
        f = conn.execute("SELECT file_path FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        fname = f[0] if f else ""
        
        print(f"Score: {score:.4f} | Len: {length} | File: {fname.split('/')[-1]}")

if __name__ == "__main__":
    main()
