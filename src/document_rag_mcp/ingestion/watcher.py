import fnmatch
from pathlib import Path
from typing import Callable, Coroutine
from watchfiles import Change, awatch
from ..config import CollectionConfig


def resolve_collection(
    file_path: Path, collections: list[CollectionConfig]
) -> tuple[CollectionConfig, Path] | None:
    """Finds the collection config and base folder that a given file path belongs to.

    Uses absolute paths so it works correctly even if the file was deleted.
    """
    abs_path = file_path.absolute()
    for coll in collections:
        for path in coll.paths:
            abs_folder = path.absolute()
            try:
                # If the configured path is directly a file
                if abs_folder.is_file():
                    if abs_path == abs_folder:
                        if any(fnmatch.fnmatch(abs_path.name, pat) for pat in coll.file_patterns):
                            return coll, abs_folder
                # If the configured path is a directory
                else:
                    if abs_path.is_relative_to(abs_folder):
                        # Ensure it's not inside a hidden folder/file
                        relative = abs_path.relative_to(abs_folder)
                        if any(part.startswith(".") for part in relative.parts):
                            continue
                        if any(fnmatch.fnmatch(abs_path.name, pat) for pat in coll.file_patterns):
                            return coll, abs_folder
            except Exception:
                pass
    return None


async def watch_collections(
    collections: list[CollectionConfig],
    callback: Callable[[str, Path, str], Coroutine[None, None, None]],
) -> None:
    """Watches configured folders recursively for added, modified, or deleted files.

    Invokes the async callback with (change_type, absolute_file_path, collection_name)
    where change_type is one of "added", "modified", "deleted".
    """
    paths_to_watch: list[Path] = []
    for coll in collections:
        for path in coll.paths:
            resolved = path.resolve()
            if resolved.exists():
                if resolved.is_file():
                    # watchfiles watches directories, so watch the parent folder of direct files
                    paths_to_watch.append(resolved.parent)
                else:
                    paths_to_watch.append(resolved)

    unique_paths = list(set(paths_to_watch))
    if not unique_paths:
        return

    # awatch yields a set of changes: (Change, path_str)
    async for changes in awatch(*unique_paths):
        for change_type, path_str in changes:
            path = Path(path_str).resolve()
            res = resolve_collection(path, collections)
            if res:
                coll, _ = res
                change_name = "modified"
                if change_type == Change.added:
                    change_name = "added"
                elif change_type == Change.deleted:
                    change_name = "deleted"

                try:
                    await callback(change_name, path, coll.name)
                except Exception as e:
                    # Log or print exception to prevent watcher from crashing completely
                    print(f"Error in watcher callback for {path}: {e}")
