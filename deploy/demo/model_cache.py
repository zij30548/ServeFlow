"""Fetch one fixed, hash-bound Chroma model at image build time.

No model API, application environment, database or arbitrary URL is used.  A
runtime check is entirely offline. Existing partial or changed destinations are
rejected rather than repaired. Chroma's six portable model files are data, not
native libraries copied from the operator's machine.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
ARCHIVE_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
ARCHIVE_LIMIT = 128 * 1024 * 1024
EXTRACTED_LIMIT = 256 * 1024 * 1024
TIMEOUT_SECONDS = 30
DOWNLOAD_DEADLINE_SECONDS = 180
FILES = {
    "config.json": "b567c7d5a55b636c95186aaf993f9a8920842b7e05a9e703e68b23cab2c3a670",
    "model.onnx": "4f148ba8ae9c2c7fbee4af2b132db8d06c6a6545b47fc83bbb98c3d22b8393e6",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "tokenizer_config.json": "7702051bbc4953b94d47fa1d61b42ed4cbb3c71b501a8dd7183a823f8bea1f20",
    "tokenizer.json": "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


class ModelCacheError(ValueError):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise ModelCacheError("model_download_redirect_rejected")


def _path(value: Path) -> Path:
    value = Path(value)
    if (not value.is_absolute() or value.name != MODEL_NAME or value.resolve() != value
            or any(parent.is_symlink() for parent in (value, *value.parents))):
        raise ModelCacheError("model_cache_path_invalid")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(destination: Path) -> None:
    """Accept exactly the six expected portable files, without downloading."""
    destination = _path(destination)
    directory = destination / "onnx"
    if (not destination.is_dir() or set(destination.iterdir()) != {directory}
            or not directory.is_dir() or directory.is_symlink()
            or {item.name for item in directory.iterdir()} != set(FILES)):
        raise ModelCacheError("model_cache_missing_or_incomplete")
    for name, expected in FILES.items():
        item = directory / name
        if (item.is_symlink() or not item.is_file() or item.stat().st_nlink != 1
                or item.stat().st_size > EXTRACTED_LIMIT or _sha(item) != expected):
            raise ModelCacheError("model_cache_file_mismatch")


def _download(path: Path, *, opener=None) -> None:
    open_url = opener or build_opener(_NoRedirect()).open
    request = Request(MODEL_URL, headers={"User-Agent": "ServeFlow-model-cache/1"})
    # Whole-download cooperative deadline, plus at most one bounded blocking
    # socket read. Container build cancellation remains the hard process bound.
    deadline = time.monotonic() + DOWNLOAD_DEADLINE_SECONDS
    try:
        with open_url(request, timeout=TIMEOUT_SECONDS) as response:
            if getattr(response, "status", None) != 200 or response.geturl() != MODEL_URL:
                raise ModelCacheError("model_download_response_invalid")
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isascii() or not length.isdecimal()
                                       or not 0 < int(length) <= ARCHIVE_LIMIT):
                raise ModelCacheError("model_archive_limit")
            total = 0
            with path.open("xb") as stream:
                while True:
                    if time.monotonic() >= deadline:
                        raise ModelCacheError("model_download_deadline")
                    block = response.read(min(64 * 1024, ARCHIVE_LIMIT - total + 1))
                    if time.monotonic() >= deadline:
                        raise ModelCacheError("model_download_deadline")
                    if not block:
                        break
                    total += len(block)
                    if total > ARCHIVE_LIMIT:
                        raise ModelCacheError("model_archive_limit")
                    stream.write(block)
            if not total or length is not None and total != int(length):
                raise ModelCacheError("model_download_incomplete")
    except (HTTPError, URLError, TimeoutError, OSError):
        raise ModelCacheError("model_download_failed") from None
    if _sha(path) != ARCHIVE_SHA256:
        raise ModelCacheError("model_archive_hash_mismatch")


def _extract(archive: Path, target: Path) -> None:
    """Write reviewed names manually; never call tar.extract or follow links."""
    expected = {"onnx/" + name for name in FILES}
    seen = set()
    total = 0
    (target / "onnx").mkdir(mode=0o755)
    try:
        with tarfile.open(archive, "r:gz") as source:
            for item in source:
                if item.isdir() and item.name.rstrip("/") == "onnx":
                    continue
                if (item.name not in expected or item.name in seen or not item.isfile()
                        or item.size < 0 or item.size > EXTRACTED_LIMIT):
                    raise ModelCacheError("model_archive_members_invalid")
                seen.add(item.name)
                total += item.size
                if total > EXTRACTED_LIMIT:
                    raise ModelCacheError("model_archive_limit")
                incoming = source.extractfile(item)
                if incoming is None:
                    raise ModelCacheError("model_archive_members_invalid")
                path = target / item.name
                with incoming, path.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=64 * 1024)
                if path.stat().st_size != item.size:
                    raise ModelCacheError("model_archive_incomplete")
                path.chmod(0o644)
    except (tarfile.TarError, EOFError, OSError):
        raise ModelCacheError("model_archive_invalid") from None
    if seen != expected:
        raise ModelCacheError("model_archive_members_invalid")


def fetch(destination: Path, *, opener=None) -> bool:
    """Return False for a valid existing cache; never overwrite a directory."""
    destination = _path(destination)
    if destination.exists():
        verify(destination)
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    _path(destination)
    # Exclusive directory creation reserves this destination. A second builder
    # sees an incomplete directory and fails; it cannot replace our work.
    try:
        destination.mkdir(mode=0o755)
    except FileExistsError:
        raise ModelCacheError("model_cache_destination_exists") from None
    with tempfile.TemporaryDirectory(prefix=".model-build-", dir=destination.parent) as raw:
        staging = Path(raw)
        archive = staging / "model.tar.gz"
        _download(archive, opener=opener)
        _extract(archive, destination)
        verify(destination)
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("fetch", "verify"))
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.operation == "fetch":
            changed = fetch(args.destination)
            print("Model cache fetched and hash-verified." if changed else "Existing model cache hash-verified; unchanged.")
        else:
            verify(args.destination)
            print("Model cache hash-verified offline.")
        return 0
    except (ModelCacheError, OSError) as error:
        code = str(error) if isinstance(error, ModelCacheError) else "model_cache_filesystem_error"
        print("STOP: " + code)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
