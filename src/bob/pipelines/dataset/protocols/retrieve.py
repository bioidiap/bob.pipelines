"""Allows to find a protocol definition file locally, or download it if needed.


Expected protocol structure:

``base_dir / subdir / database_filename / protocol_name / group_name``


By default, ``base_dir`` will be pointed by the ``bob_data_dir`` config.
``subdir`` is provided as a way to use a directory inside ``base_dir`` when
using its default.

Here are some valid example paths (``bob_data_dir=/home/username/bob_data``):

In a "raw" directory (not an archive):

``/home/username/bob_data/protocols/my_db/my_protocol/my_group``

In an archive:

``/home/username/bob_data/protocols/my_db.tar.gz/my_protocol/my_group``

In an archive with the database name as top-level (some legacy db used that):

``/home/username/bob_data/protocols/my_db.tar.gz/my_db/my_protocol/my_group``

"""

import glob

from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Optional, TextIO, Union

import requests

from clapper.rc import UserDefaults

from bob.pipelines.dataset.protocols import archive, hashing

logger = getLogger(__name__)


def _get_local_data_directory() -> Path:
    """Returns the local directory for data (``bob_data_dir`` config)."""
    user_config = UserDefaults("bobrc.toml")
    return Path(
        user_config.get("bob_data_dir", default=Path.home() / "bob_data")
    )


def _infer_filename_from_urls(urls=Union[list[str], str]) -> str:
    """Retrieves the remote filename from the URLs.

    Parameters
    ----------
    urls
        One or multiple URLs pointing to files with the same name.

    Returns
    -------
    The remote file name.

    Raises
    ------
    ValueError
        When urls point to files with different names.
    """
    if isinstance(urls, str):
        return urls.split("/")[-1]

    # Check that all urls point to the same file name
    names = [u.split("/")[-1] for u in urls]
    if not all(n == names[0] for n in names):
        raise ValueError(
            f"Cannot infer file name when urls point to different files ({names=})."
        )
    return urls[0].split("/")[-1]


def retrieve_protocols(
    urls: list[str],
    destination_filename: Optional[str] = None,
    base_dir: Union[PathLike[str], str, None] = None,
    subdir: Union[PathLike[str], str] = "protocol",
    checksum: Union[str, None] = None,
) -> Path:
    """Automatically downloads the necessary protocol definition files."""
    if base_dir is None:
        base_dir = _get_local_data_directory()

    remote_filename = _infer_filename_from_urls(urls)
    if destination_filename is None:
        destination_filename = remote_filename
    elif Path(remote_filename).suffixes != Path(destination_filename).suffixes:
        raise ValueError(
            "Local dataset protocol definition files must have the same "
            f"extension as the remote ones ({remote_filename=})"
        )

    return download_protocol_definition(
        urls=urls,
        destination_base_dir=base_dir,
        destination_subdir=subdir,
        destination_filename=destination_filename,
        checksum=checksum,
        force=False,
    )


def list_protocol_paths(
    database_name: str,
    base_dir: Union[PathLike[str], str, None] = None,
    subdir: Union[PathLike[str], str] = "protocol",
    database_filename: Union[str, None] = None,
) -> list[Path]:
    """Returns the paths of each protocol in a database definition file."""
    if base_dir is None:
        base_dir = _get_local_data_directory()
    final_dir = Path(base_dir) / subdir
    final_dir /= (
        database_name if database_filename is None else database_filename
    )

    if archive.is_archive(final_dir):
        protocols = archive.list_dirs(final_dir, show_files=False)
        if len(protocols) == 1 and protocols[0].name == database_name:
            protocols = archive.list_dirs(
                final_dir, database_name, show_files=False
            )

        archive_path, inner_dir = archive.path_and_subdir(final_dir)
        if inner_dir is None:
            return [
                Path(f"{archive_path.as_posix()}:{p.as_posix().lstrip('/')}")
                for p in protocols
            ]

        return [
            Path(f"{archive_path.as_posix()}:{inner_dir.as_posix()}/{p.name}")
            for p in protocols
        ]

    # Not an archive
    return final_dir.iterdir()


def get_protocol_path(
    database_name: str,
    protocol: str,
    base_dir: Union[PathLike[str], str, None] = None,
    subdir: Union[PathLike[str], str] = "protocols",
    database_filename: Optional[str] = None,
) -> Union[Path, None]:
    """Returns the path of a specific protocol.

    Will look for ``protocol`` in ``base_dir / subdir / database_(file)name``.

    Returns
    -------
    Path
        The required protocol's path for the database.
    """
    protocol_paths = list_protocol_paths(
        database_name=database_name,
        base_dir=base_dir,
        subdir=subdir,
        database_filename=database_filename,
    )
    for protocol_path in protocol_paths:
        if archive.is_archive(protocol_path):
            _base, inner = archive.path_and_subdir(protocol_path)
            if inner.name == protocol:
                return protocol_path
        elif protocol_path.name == protocol:
            return protocol_path
    logger.warning(f"Protocol {protocol} not found in {database_name}.")
    return None


def list_protocol_names(
    database_name: str,
    base_dir: Union[PathLike[str], str, None] = None,
    subdir: Union[PathLike[str], str, None] = "protocols",
    database_filename: Union[str, None] = None,
) -> list[str]:
    """Returns the paths of the protocol directories for a given database.

    Archives are also accepted, either if the file name is the same as
    ``database_name`` with a ``.tar.gz`` extension or by specifying the filename
    in ``database_filename``.

    This will look in ``base_dir/subdir`` for ``database_filename``, then
    ``database_name``, then ``database_name+".tar.gz"``.

    Parameters
    ----------
    database_name
        The database name used to infer ``database_filename`` if not specified.
    base_dir
        The base path of data files (defaults to the ``bob_data_dir`` config, or
        ``~/bob_data`` if not configured).
    subdir
        A sub directory for the protocols in ``base_dir``.
    database_filename
        If the file/directory name of the protocols is not the same as the
        name of the database, this can be set to look in the correct file.

    Returns
    -------
    A list of protocol names
        The different protocols available for that database.
    """

    if base_dir is None:
        base_dir = _get_local_data_directory()

    if subdir is None:
        subdir = "."

    if database_filename is None:
        database_filename = database_name
        final_path: Path = Path(base_dir) / subdir / database_filename
        if not final_path.is_dir():
            database_filename = database_name + ".tar.gz"

    final_path: Path = Path(base_dir) / subdir / database_filename

    if archive.is_archive(final_path):
        top_level_dirs = archive.list_dirs(final_path, show_files=False)
        # Handle a database archive having database_name as top-level directory
        if len(top_level_dirs) == 1 and top_level_dirs[0].name == database_name:
            return [
                p.name
                for p in archive.list_dirs(
                    final_path, inner_dir=database_name, show_files=False
                )
            ]
        return [p.name for p in top_level_dirs]
    # Not an archive: list the dirs
    return [p.name for p in final_path.iterdir() if p.is_dir()]


def open_definition_file(
    search_pattern: Union[PathLike[str], str],
    database_name: str,
    protocol: str,
    base_dir: Union[PathLike[str], str, None] = None,
    subdir: Union[PathLike[str], str, None] = "protocols",
    database_filename: Optional[str] = None,
) -> Union[TextIO, None]:
    """Opens a protocol definition file inside a protocol directory.

    Also handles protocols inside an archive.
    """
    search_path = get_protocol_path(
        database_name, protocol, base_dir, subdir, database_filename
    )

    if archive.is_archive(search_path):
        return archive.search_and_open(
            search_pattern=search_pattern,
            archive_path=search_path,
        )

    search_pattern = Path(search_pattern)

    # we prepend './' to search_pattern because it might start with '/'
    pattern = search_path / "**" / f"./{search_pattern.as_posix()}"
    for path in glob.iglob(pattern.as_posix(), recursive=True):
        if not Path(path).is_file():
            continue
        return open(path, mode="rt")
    logger.info(f"Unable to locate and open a file that matches '{pattern}'.")
    return None


def list_group_paths(
    database_name: str,
    protocol: str,
    base_dir: Union[PathLike[str], str, None] = None,
    subdir: Union[PathLike[str], str] = "protocols",
    database_filename: Optional[str] = None,
) -> list[Path]:
    """Returns the file paths of the groups in protocol"""
    protocol_path = get_protocol_path(
        database_name=database_name,
        protocol=protocol,
        base_dir=base_dir,
        subdir=subdir,
        database_filename=database_filename,
    )
    if archive.is_archive(protocol_path):
        groups_inner = archive.list_dirs(protocol_path)
        archive_path, inner_path = archive.path_and_subdir(protocol_path)
        return [
            Path(f"{archive_path.as_posix()}:{inner_path.as_posix()}/{g}")
            for g in groups_inner
        ]
    return protocol_path.iterdir()


def list_group_names(
    database_name: str,
    protocol: str,
    base_dir: Union[PathLike[str], str, None] = None,
    subdir: Union[PathLike[str], str] = "protocols",
    database_filename: Optional[str] = None,
) -> list[str]:
    """Returns the group names of a protocol."""
    paths = list_group_paths(
        database_name=database_name,
        protocol=protocol,
        base_dir=base_dir,
        subdir=subdir,
        database_filename=database_filename,
    )
    # Supports groups as files or dirs
    return [p.stem for p in paths]  # ! This means group can't include a '.'


def download_protocol_definition(
    urls: Union[list[str], str],
    destination_base_dir: Union[PathLike, None] = None,
    destination_subdir: Union[str, None] = None,
    destination_filename: Union[str, None] = None,
    checksum: Union[str, None] = None,
    checksum_fct: Callable[[Any, int], str] = hashing.sha256_hash,
    force: bool = False,
    makedirs: bool = True,
) -> Path:
    """Downloads a remote file locally.

    Parameters
    ----------
    urls
        The remote location of the server. If multiple addresses are given, we will try
        to download from them in order until one succeeds.
    destination_basedir
        A path to a local directory where the file will be saved. If omitted, the file
        will be saved in the folder pointed by the ``wdr.local_directory`` key in the
        user configuration.
    destination_subdir
        An additional layer added to the destination directory (useful when using
        ``destination_directory=None``).
    destination_filename
        The final name of the local file. If omitted, the file will keep the name of
        the remote file.
    checksum
        When provided, will compute the file's checksum and compare to this.
    checksum_fct
        A callable that takes a ``reader`` and returns a hash.
    force
        Re-download and overwrite any existing file with the same name.
    makedirs
        Automatically make the parent directories of the new local file.

    Returns
    -------
    The path to the new local file.

    Raises
    ------
    RuntimeError
        When the URLs provided are all invalid.
    ValueError
        When ``destination_filename`` is omitted and URLs point to files with different
        names.
        When the checksum of the file does not correspond to the provided ``checksum``.
    """

    if destination_filename is None:
        destination_filename = _infer_filename_from_urls(urls=urls)

    if destination_base_dir is None:
        destination_base_dir = _get_local_data_directory()

    destination_base_dir = Path(destination_base_dir)

    if destination_subdir is not None:
        destination_base_dir = destination_base_dir / destination_subdir

    local_file = destination_base_dir / destination_filename
    needs_download = True

    if not force and local_file.is_file():
        if checksum is None:
            logger.info(
                f"File {local_file} already exists, skipping download ({force=})."
            )
            needs_download = False
        elif hashing.verify_file(local_file, checksum, checksum_fct):
            logger.info(
                f"File {local_file} already exists and checksum is valid."
            )
            needs_download = False

    if needs_download:
        if isinstance(urls, str):
            urls = [urls]

        for tries, url in enumerate(urls):
            logger.debug(f"Retrieving file from '{url}'.")
            try:
                response = requests.get(url=url, timeout=10)
            except requests.exceptions.ConnectionError as e:
                if tries < len(urls) - 1:
                    logger.info(
                        f"Could not connect to {url}. Trying other URLs."
                    )
                logger.debug(e)
                continue

            logger.debug(
                f"http response: '{response.status_code}: {response.reason}'."
            )

            if response.ok:
                logger.debug(f"Got file from {url}.")
                break
            if tries < len(urls) - 1:
                logger.info(
                    f"Failed to get file from {url}, trying other URLs."
                )
                logger.debug(f"requests.response was\n{response}")
        else:
            raise RuntimeError(
                f"Could not retrieve file from any of the provided URLs! ({urls=})"
            )

        if makedirs:
            local_file.parent.mkdir(parents=True, exist_ok=True)

        with local_file.open("wb") as f:
            f.write(response.content)

    if checksum is not None:
        if not hashing.verify_file(local_file, checksum, hash_fct=checksum_fct):
            if not needs_download:
                raise ValueError(
                    f"The local file hash does not correspond to '{checksum}' "
                    f"and {force=} prevents overwriting."
                )
            raise ValueError(
                "The downloaded file hash ('"
                f"{hashing.compute_crc(local_file, hash_fct=checksum_fct)}') does "
                f"not correspond to '{checksum}'."
            )

    return local_file
