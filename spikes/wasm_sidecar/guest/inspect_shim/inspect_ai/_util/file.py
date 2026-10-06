"""Stand-in: the real module pulls fsspec/s3fs; the guest has no filesystem."""

import os


def exists(path: str) -> bool:
    return os.path.exists(path)


def local_path(path: str) -> str:
    return path.removeprefix("file://")
