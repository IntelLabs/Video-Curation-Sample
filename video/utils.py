import os
import re

def safely_join_path(base_dir, add_path):
    safe_base = os.path.abspath(base_dir)
    candidate_path = os.path.abspath(os.path.join(safe_base, add_path))
    if not candidate_path.startswith(safe_base + os.sep):
        raise ValueError(f"Invalid path: {candidate_path}")
    return candidate_path


def str2bool(in_val):
    if isinstance(in_val, bool):
        return in_val

    if not isinstance(in_val, str):
        raise ValueError(f"{in_val} is not a bool or string")

    if in_val.title() == "True":
        return True
    else:
        return False


def validate_video_name(name):
    """
    Validate that the provided video identifier is a simple file name and
    not a path. Returns a normalized name or raises ValueError.
    """
    if not isinstance(name, str):
        raise ValueError("Video name must be a string")
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Video name cannot be empty")
    # Disallow path separators to ensure this is just a file name.
    if os.sep in cleaned or "/" in cleaned or "\\" in cleaned:
    # Restrict the video name to a safe subset of characters to avoid
    # passing arbitrary strings to external commands.
    # Allow letters, digits, underscore, hyphen and dot, and disallow
    # leading dot to avoid hidden or special files.
    if cleaned.startswith("."):
        raise ValueError(f"Invalid video name: {cleaned}")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", cleaned):
        raise ValueError(f"Invalid video name: {cleaned}")
        raise ValueError(f"Invalid video name: {cleaned}")
    return cleaned
