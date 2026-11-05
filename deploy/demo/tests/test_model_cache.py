"""Portable cache integrity tests; fake archives and no network or model API."""
import ast
import hashlib
import importlib.util
import io
from pathlib import Path
import socket
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("demo_model_cache", ROOT / "deploy/demo/model_cache.py")
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("cache tests must not contact a service")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


def archive(rows):
    target = io.BytesIO()
    with tarfile.open(fileobj=target, mode="w:gz") as writer:
        for name, data, kind in rows:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(data) if kind == tarfile.REGTYPE else 0
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                info.linkname = "../../outside"
            writer.addfile(info, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
    return target.getvalue()


class Response(io.BytesIO):
    def __init__(self, data, *, length="auto", url=cache.MODEL_URL):
        super().__init__(data)
        self.status, self.url = 200, url
        self.headers = {} if length is None else {"Content-Length": str(len(data)) if length == "auto" else length}

    def geturl(self):
        return self.url


@pytest.fixture
def model(tmp_path, monkeypatch):
    contents = {"config.json": b"controlled-config", "model.onnx": b"controlled-portable-model"}
    monkeypatch.setattr(cache, "FILES", {name: hashlib.sha256(raw).hexdigest() for name, raw in contents.items()})
    raw = archive([("onnx/" + name, data, tarfile.REGTYPE) for name, data in contents.items()])
    monkeypatch.setattr(cache, "ARCHIVE_SHA256", hashlib.sha256(raw).hexdigest())
    destination = tmp_path.resolve() / cache.MODEL_NAME
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        return Response(raw)

    return destination, contents, raw, calls, opener


def test_portable_file_hashes_match_existing_frozen_data_contract():
    tree = ast.parse((ROOT / "backend/evaluation/task_data.py").read_text())
    expected = next(ast.literal_eval(row.value) for row in tree.body if isinstance(row, ast.Assign)
                    and any(isinstance(name, ast.Name) and name.id == "MODEL_FILES" for name in row.targets))
    assert cache.FILES == expected
    assert len(cache.FILES) == 6 and all(len(value) == 64 for value in cache.FILES.values())
    assert cache.MODEL_URL == "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
    assert cache.ARCHIVE_SHA256 == "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"


def test_fetch_verify_and_second_build_do_not_overwrite_or_download_again(model):
    target, contents, _, calls, opener = model
    assert cache.fetch(target, opener=opener) is True
    cache.verify(target)
    assert cache.fetch(target, opener=lambda *a, **k: pytest.fail("existing cache downloaded")) is False
    assert calls == [(cache.MODEL_URL, 30)]
    assert {p.name: p.read_bytes() for p in (target / "onnx").iterdir()} == contents
    assert not list(target.parent.glob(".model-build-*"))


def test_changed_or_partial_cache_is_preserved_without_repair(model):
    target, _, _, _, opener = model
    cache.fetch(target, opener=opener)
    changed = target / "onnx/model.onnx"
    changed.write_bytes(b"changed")
    with pytest.raises(cache.ModelCacheError, match="mismatch"):
        cache.fetch(target, opener=lambda *a, **k: pytest.fail("corrupt cache downloaded"))
    assert changed.read_bytes() == b"changed"


@pytest.mark.parametrize("damage", ["symlink", "hardlink", "extra", "missing"])
def test_verify_rejects_unexpected_or_referenced_files(model, damage):
    target, _, _, _, opener = model
    cache.fetch(target, opener=opener)
    item = target / "onnx/model.onnx"
    if damage == "symlink":
        saved = target.parent / "elsewhere"
        saved.write_bytes(item.read_bytes())
        item.unlink()
        item.symlink_to(saved)
    elif damage == "hardlink":
        import os
        os.link(item, target.parent / "other-link")
    elif damage == "extra":
        (target / "onnx/unknown").write_bytes(b"unknown")
    else:
        item.unlink()
    with pytest.raises(cache.ModelCacheError):
        cache.verify(target)


@pytest.mark.parametrize("member,kind", [
    ("../escape", tarfile.REGTYPE), ("/tmp/escape", tarfile.REGTYPE),
    ("onnx/model.onnx", tarfile.SYMTYPE), ("onnx/model.onnx", tarfile.LNKTYPE),
    ("onnx/unknown.json", tarfile.REGTYPE),
])
def test_archive_paths_links_and_unknown_members_are_rejected(model, monkeypatch, member, kind):
    target, _, _, _, _ = model
    raw = archive([(member, b"unsafe", kind)])
    monkeypatch.setattr(cache, "ARCHIVE_SHA256", hashlib.sha256(raw).hexdigest())
    with pytest.raises(cache.ModelCacheError, match="members_invalid"):
        cache.fetch(target, opener=lambda *a, **k: Response(raw))
    assert not (target.parent / "escape").exists()
    assert not list(target.parent.glob(".model-build-*"))


def test_duplicate_member_is_not_silently_overwritten(model, monkeypatch):
    target, _, _, _, _ = model
    raw = archive([("onnx/config.json", b"first", tarfile.REGTYPE),
                   ("onnx/config.json", b"second", tarfile.REGTYPE)])
    monkeypatch.setattr(cache, "ARCHIVE_SHA256", hashlib.sha256(raw).hexdigest())
    with pytest.raises(cache.ModelCacheError, match="members_invalid"):
        cache.fetch(target, opener=lambda *a, **k: Response(raw))
    assert (target / "onnx/config.json").read_bytes() == b"first"


def test_wrong_archive_hash_never_extracts(model):
    target, _, _, _, _ = model
    with pytest.raises(cache.ModelCacheError, match="hash_mismatch"):
        cache.fetch(target, opener=lambda *a, **k: Response(b"not-the-archive"))
    assert not (target / "onnx").exists()


@pytest.mark.parametrize("length", ["0", "invalid", "9999999999999", "-1"])
def test_untrusted_content_length_cannot_expand_limits(model, length):
    target, _, raw, _, _ = model
    with pytest.raises(cache.ModelCacheError, match="archive_limit"):
        cache.fetch(target, opener=lambda *a, **k: Response(raw, length=length))


def test_stream_limit_applies_without_content_length(model, monkeypatch):
    target, _, raw, _, _ = model
    monkeypatch.setattr(cache, "ARCHIVE_LIMIT", 20)
    with pytest.raises(cache.ModelCacheError, match="archive_limit"):
        cache.fetch(target, opener=lambda *a, **k: Response(raw, length=None))


def test_incomplete_body_is_not_accepted(model):
    target, _, raw, _, _ = model
    with pytest.raises(cache.ModelCacheError, match="incomplete"):
        cache.fetch(target, opener=lambda *a, **k: Response(raw, length=str(len(raw) + 1)))


def test_slow_download_has_a_whole_operation_deadline(model, monkeypatch):
    target, _, raw, _, _ = model
    clock = iter([0.0, 0.0, cache.DOWNLOAD_DEADLINE_SECONDS + 1.0])
    monkeypatch.setattr(cache.time, "monotonic", lambda: next(clock))
    with pytest.raises(cache.ModelCacheError, match="deadline"):
        cache.fetch(target, opener=lambda *a, **k: Response(raw))


def test_redirect_and_nonfixed_final_url_are_rejected(model):
    target, _, raw, _, _ = model
    with pytest.raises(cache.ModelCacheError, match="response_invalid"):
        cache.fetch(target, opener=lambda *a, **k: Response(raw, url="https://example.invalid/model"))
    with pytest.raises(cache.ModelCacheError, match="redirect_rejected"):
        cache._NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://example.invalid")


def test_symlink_destination_or_ancestor_never_downloads(model):
    target, _, _, _, _ = model
    other = target.parent / "other"
    other.mkdir()
    target.symlink_to(other, target_is_directory=True)
    with pytest.raises(cache.ModelCacheError, match="path_invalid"):
        cache.fetch(target, opener=lambda *a, **k: pytest.fail("unsafe path downloaded"))


def test_extracted_size_limit_is_independent_from_archive_limit(model, monkeypatch):
    target, _, raw, _, _ = model
    monkeypatch.setattr(cache, "EXTRACTED_LIMIT", 1)
    with pytest.raises(cache.ModelCacheError):
        cache.fetch(target, opener=lambda *a, **k: Response(raw))


def test_cli_has_no_arbitrary_url_or_expected_hash_options(model):
    with pytest.raises(SystemExit) as result:
        cache.main(["fetch", str(model[0]), "--url", "https://example.invalid/model"])
    assert result.value.code == 2
