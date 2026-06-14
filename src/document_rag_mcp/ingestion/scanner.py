import fnmatch
from pathlib import Path
from ..config import CollectionConfig


def scan_collection_files(collection: CollectionConfig) -> list[Path]:
    """Scans all configured paths in a collection recursively for matching files."""
    matched_files: list[Path] = []
    for path in collection.paths:
        if not path.exists():
            continue

        if path.is_file():
            # If path points directly to a file, verify it matches the patterns
            if any(fnmatch.fnmatch(path.name, pat) for pat in collection.file_patterns):
                matched_files.append(path.resolve())
        elif path.is_dir():
            # If path is a directory, scan recursively
            for p in path.rglob("*"):
                # Ignore hidden directories/files (like .git, .mcp, etc.)
                if any(part.startswith(".") for part in p.relative_to(path).parts):
                    continue
                if p.is_file():
                    if any(fnmatch.fnmatch(p.name, pat) for pat in collection.file_patterns):
                        matched_files.append(p.resolve())

    return matched_files


def scan_all_collections(collections: list[CollectionConfig]) -> dict[str, list[Path]]:
    """Scans all collections and returns a dictionary mapping collection names to lists of absolute file paths."""
    results: dict[str, list[Path]] = {}
    for coll in collections:
        results[coll.name] = scan_collection_files(coll)
    return results
