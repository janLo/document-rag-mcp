import chonkie
import inspect

print("Chunkers in chonkie:")
for name, obj in inspect.getmembers(chonkie):
    if inspect.isclass(obj) and "Chunker" in name:
        print(f" - {name}")

print("\nSemanticChunker init args:")
print(inspect.signature(chonkie.SemanticChunker.__init__))

print("\nRecursiveChunker init args:")
print(inspect.signature(chonkie.RecursiveChunker.__init__))

print("\nTokenChunker init args:")
print(inspect.signature(chonkie.TokenChunker.__init__))
