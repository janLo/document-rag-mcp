import chonkie
from chonkie import SemanticChunker

text = """
# Anlage 7: Personalrichtwerte

Hier ist eine Tabelle mit Werten.
Und noch ein Satz zur Erklärung.
"""

chunker1 = SemanticChunker(
    embedding_model="all-MiniLM-L6-v2",
    threshold=0.5,
    chunk_size=512,
    min_sentences_per_chunk=1
)

chunker2 = SemanticChunker(
    embedding_model="all-MiniLM-L6-v2",
    threshold=0.5,
    chunk_size=512,
    min_sentences_per_chunk=2
)

print("--- min_sentences_per_chunk = 1 ---")
for i, c in enumerate(chunker1.chunk(text)):
    print(f"[{i}] {repr(c.text)}")

print("\n--- min_sentences_per_chunk = 2 ---")
for i, c in enumerate(chunker2.chunk(text)):
    print(f"[{i}] {repr(c.text)}")

