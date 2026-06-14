import os
from pathlib import Path
import tempfile

# Force a temporary data directory for all tests to avoid polluting the workspace
test_temp_dir = tempfile.TemporaryDirectory()
os.environ["DOCRAG_STORAGE__DATA_DIR"] = str(Path(test_temp_dir.name) / "data")
