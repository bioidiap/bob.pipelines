"""Archives (tar, zip) operations like searching for files and extracting."""

import bz2
import io
import logging
import os
import tarfile
import zipfile

from fnmatch import fnmatch
from pathlib import Path
from typing import IO, TextIO, Union

logger = logging.getLogger(__name__)


def path_and_subdir(
    archive_path: Union[str, os.PathLike],
) -> tuple[Path, Union[Path, None]]:
    """Splits an archive's path from a sub directory (separated by ``:``)."""
    archive_path_str = Path(archive_path).as_posix()
    if ":" in archive_path_str:
        archive, sub_dir = archive_path_str.rsplit(":", 1)
        return Path(archive), Path(sub_dir)
    return Path(archive_path), None


def _is_bz2(path: Union[str, os.PathLike]) -> bool:
    try:
        with bz2.BZ2File(path) as f:
            f.read(1024)
        return True
    except (OSError, EOFError):
        return False


def is_archive(path: Union[str, os.PathLike]) -> bool:
    """Returns whether the path points in an archive.

    Any path pointing to a valid tar or zip archive or to a valid bz2
    file will return ``True``.
    """
    archive = path_and_subdir(path)[0]
    try:
        return any(
            tester(path_and_subdir(archive)[0])
            for tester in (tarfile.is_tarfile, zipfile.is_zipfile, _is_bz2)
        )
    except (FileNotFoundError, IsADirectoryError):
        return False


def search_and_open(
    search_pattern: str,
    archive_path: Union[str, os.PathLike],
    inner_dir: Union[os.PathLike, None] = None,
    open_as_binary: bool = False,
) -> Union[IO[bytes], TextIO, None]:
    """Returns a read-only stream of a file matching a pattern in an archive.

    Wildcards (``*``, ``?``, and ``**``) are supported (using
    :meth:`pathlib.Path.glob`).

    The first matching file will be open and returned.

    examples:

    .. code-block: text

        archive.tar.gz
            + subdir1
            |   + file1.txt
            |   + file2.txt
            |
            + subdir2
                + file1.txt

    ``search_and_open("archive.tar.gz", "file1.txt")``
    opens``archive.tar.gz/subdir1/file1.txt``

    ``search_and_open("archive.tar.gz:subdir2", "file1.txt")``
    opens ``archive.tar.gz/subdir2/file1.txt``

    ``search_and_open("archive.tar.gz", "*.txt")``
    opens ``archive.tar.gz/subdir1/file1.txt``


    Parameters
    ----------
    archive_path
        The ``.tar.gz`` archive file containing the wanted file. To match
        ``search_pattern`` in a sub path in that archive, append the sub path
        to ``archive_path`` with a ``:`` (e.g.
        ``/path/to/archive.tar.gz:sub/dir/``).
    search_pattern
        A string to match to the file. Wildcards are supported (Unix pattern
        matching).

    Returns
    -------
    io.TextIOBase or io.BytesIO
        A read-only file stream.
    """

    archive_path = Path(archive_path)

    if inner_dir is None:
        archive_path, inner_dir = path_and_subdir(archive_path)

    if inner_dir is not None:
        pattern = (Path("/") / inner_dir / search_pattern).as_posix()
    else:
        pattern = (Path("/") / search_pattern).as_posix()

    if ".tar" in archive_path.suffixes:
        tar_arch = tarfile.open(archive_path)  # TODO File not closed
        for member in tar_arch:
            if member.isfile() and fnmatch("/" + member.name, pattern):
                break
        else:
            logger.debug(
                f"No file matching '{pattern}' were found in '{archive_path}'."
            )
            return None

        if open_as_binary:
            return tar_arch.extractfile(member)
        return io.TextIOWrapper(tar_arch.extractfile(member), encoding="utf-8")

    elif archive_path.suffix == ".zip":
        zip_arch = zipfile.ZipFile(archive_path)
        for name in zip_arch.namelist():
            if fnmatch("/" + name, pattern):
                break
        else:
            logger.debug(
                f"No file matching '{pattern}' were found in '{archive_path}'."
            )
        return zip_arch.open(name)

    raise ValueError(
        f"Unknown file extension '{''.join(archive_path.suffixes)}'"
    )


def list_dirs(
    archive_path: Union[str, os.PathLike],
    inner_dir: Union[os.PathLike, None] = None,
    show_dirs: bool = True,
    show_files: bool = True,
) -> list[Path]:
    """Returns a list of all the elements in an archive or inner directory.

    Parameters
    ----------
    archive_path
        A path to an archive, or an inner directory of an archive (appended
        with a ``:``).
    inner_dir
        A path inside the archive with its root at the archive's root.
    show_dirs
        Returns directories.
    show_files
        Returns files.
    """

    archive_path, arch_inner_dir = path_and_subdir(archive_path)
    inner_dir = Path(inner_dir or arch_inner_dir or Path("."))

    results = []
    # Read the archive info and iterate over the paths. Return the ones we want.
    if ".tar" in archive_path.suffixes:
        with tarfile.open(archive_path) as arch:
            for info in arch.getmembers():
                path = Path(info.name)
                if path.parent != inner_dir:
                    continue
                if info.isdir() and show_dirs:
                    results.append(Path("/") / path)
                if info.isfile() and show_files:
                    results.append(Path("/") / path)
    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as arch:
            for zip_info in arch.infolist():
                zip_path = zipfile.Path(archive_path, zip_info.filename)
                if Path(zip_info.filename).parent != inner_dir:
                    continue
                if zip_path.is_dir() and show_dirs:
                    results.append(Path("/") / zip_info.filename)
                if not zip_path.is_dir() and show_files:
                    results.append(Path("/") / zip_info.filename)
    elif archive_path.suffix == ".bz2":
        if inner_dir != Path("."):
            raise ValueError(
                ".bz2 files don't have an inner structure (tried to access "
                f"'{archive_path}:{inner_dir}')."
            )
        results.extend([Path(archive_path.stem)] if show_files else [])
    else:
        raise ValueError(
            f"Unsupported archive extension '{''.join(archive_path.suffixes)}'."
        )
    return sorted(results)  # Fixes inconsistent file ordering across platforms
