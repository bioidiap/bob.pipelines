"""Hashing functionalities for verifying files and computing CRCs."""


import hashlib
import os

from pathlib import Path
from typing import Any, Callable, Union


def md5_hash(readable: Any, chunk_size: int = 65535) -> str:
    """Computes the md5 hash of any object with a read method."""
    hasher = hashlib.md5()
    for chunk in iter(lambda: readable.read(chunk_size), b""):
        hasher.update(chunk)
    return hasher.hexdigest()


def sha256_hash(readable: Any, chunk_size: int = 65535) -> str:
    """Computes the SHA256 hash of any object with a read method."""
    hasher = hashlib.sha256()
    for chunk in iter(lambda: readable.read(chunk_size), b""):
        hasher.update(chunk)
    return hasher.hexdigest()


def verify_file(
    file_path: Union[str, os.PathLike],
    file_hash: str,
    hash_fct: Callable[[Any, int], str] = sha256_hash,
    full_match: bool = False,
) -> bool:
    """Returns True if the file computed hash corresponds to `file_hash`.

    For comfort, we allow ``file_hash`` to match with the first
    characters of the digest, allowing storing only e.g. the first 8
    char.

    Parameters
    ----------
    file_path
        The path to the file needing verification.
    file_hash
        The expected file hash digest.
    hash_fct
        A function taking a path and returning a digest. Defaults to SHA256.
    full_match
        If set to False, allows ``file_hash`` to match the first characters of
        the files digest (this allows storing e.g. 8 chars of a digest instead
        of the whole 64 characters of SHA256, and still matching.)
    """
    file_path = Path(file_path)
    with file_path.open("rb") as f:
        digest = hash_fct(f, 65535)
    return digest == file_hash if full_match else digest.startswith(file_hash)


def compute_crc(
    file_path: Union[str, os.PathLike],
    hash_fct: Callable[[Any, int], str] = sha256_hash,
) -> str:
    """Returns the CRC of a file."""
    file_path = Path(file_path)
    with file_path.open("rb") as f:
        return hash_fct(f, 65535)
