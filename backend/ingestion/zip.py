import os
import stat
import zipfile
from . import REPO_DIR
from .github import clear_repo_dir

MAX_ARCHIVE_FILES = 10_000
MAX_UNCOMPRESSED_BYTES = 500 * 1024 * 1024


def is_safe_path(basedir, path):
    """Check containment without the prefix collision in startswith checks."""
    try:
        return os.path.commonpath([os.path.realpath(basedir), os.path.realpath(path)]) == os.path.realpath(basedir)
    except ValueError:
        return False

def extract_local_zip(zip_path: str):
    """Extracts a local ZIP file to the REPO_DIR securely."""
    clear_repo_dir()
    print(f"Extracting {zip_path} into {REPO_DIR}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        if len(members) > MAX_ARCHIVE_FILES:
            raise ValueError(f"ZIP archive contains more than {MAX_ARCHIVE_FILES} files.")
        if sum(member.file_size for member in members) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("ZIP archive expands beyond the 500 MB safety limit.")

        for member in members:
            member_path = os.path.join(REPO_DIR, member.filename)
            if not is_safe_path(REPO_DIR, member_path):
                raise ValueError("ZIP archive contains a path outside the extraction directory.")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("ZIP archive contains unsupported symbolic links.")
        zip_ref.extractall(REPO_DIR)
    print("Extraction complete.")
