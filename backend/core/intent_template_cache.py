"""Private immutable Qwen template versions; readers never initialize or send.

Preparation is explicit, single-writer and bounded. An immutable per-batch claim
precedes the possible send. A claim without a validated result blocks re-sending
that batch after failure/restart. Protected deployments also use their existing
global model-send ledger in qwen_embedding. No credential or template text is
stored here; vectors remain private application data.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
try:
    import fcntl
except ImportError:
    fcntl = None
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import re

from core import qwen_embedding as qwen

VERSION = 1
MAX_TEMPLATES = 256
FILE_LIMIT = 8 * 1024 * 1024
PREPARE_DEADLINE = 60.0
ROOT_MARKER = {'version': VERSION, 'kind': 'serveflow-qwen-intent-templates'}


class TemplateCacheError(ValueError):
    CODES = frozenset({'embedding_templates_not_prepared', 'embedding_template_cache_invalid',
        'embedding_template_preparation_busy', 'embedding_template_preparation_incomplete',
        'embedding_template_limit', 'embedding_template_request_limit', 'embedding_template_prepare_timeout'})

    def __init__(self, code='embedding_template_cache_invalid'):
        self.code = code if code in self.CODES else 'embedding_template_cache_invalid'
        super().__init__(self.code)


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()


def _same(value, expected):
    return type(value) is dict and canonical(value) == canonical(expected)


def _file(fd):
    s = os.fstat(fd)
    if (not stat.S_ISREG(s.st_mode) or s.st_uid != os.geteuid()
            or stat.S_IMODE(s.st_mode) != 0o600 or s.st_nlink != 1):
        raise TemplateCacheError()


def _directory(path, *, parent_fd=None, create=False):
    try:
        if create:
            try: os.mkdir(path, 0o700, dir_fd=parent_fd)
            except FileExistsError: pass
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        s = os.fstat(fd)
        if s.st_uid != os.geteuid() or stat.S_IMODE(s.st_mode) != 0o700:
            os.close(fd)
            raise TemplateCacheError()
        return fd
    except FileNotFoundError:
        raise TemplateCacheError('embedding_templates_not_prepared') from None


def _read(directory, name, *, maximum=FILE_LIMIT):
    fd = None
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        _file(fd)
        size = os.fstat(fd).st_size
        if not 0 < size <= maximum: raise TemplateCacheError()
        chunks, count = [], 0
        while True:
            part = os.read(fd, min(65536, maximum + 1 - count))
            if not part: break
            count += len(part)
            if count > maximum: raise TemplateCacheError()
            chunks.append(part)
        if count != size: raise TemplateCacheError()
        now = os.stat(name, dir_fd=directory, follow_symlinks=False)
        opened = os.fstat(fd)
        if (now.st_ino, now.st_dev) != (opened.st_ino, opened.st_dev): raise TemplateCacheError()
        return json.loads(b''.join(chunks), object_pairs_hook=qwen._pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except FileNotFoundError:
        raise TemplateCacheError('embedding_templates_not_prepared') from None
    except (ValueError, TypeError, OSError, RecursionError):
        raise TemplateCacheError() from None
    finally:
        if fd is not None: os.close(fd)


def _exists(directory, name):
    try: os.stat(name, dir_fd=directory, follow_symlinks=False); return True
    except FileNotFoundError: return False


def _write(directory, name, value):
    """Exclusive claim/identity, durable before any possible send."""
    raw = canonical(value) + b'\n'
    if len(raw) > FILE_LIMIT: raise TemplateCacheError()
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    try:
        view = memoryview(raw)
        while view:
            n = os.write(fd, view)
            if n <= 0: raise TemplateCacheError()
            view = view[n:]
        os.fsync(fd)
    finally: os.close(fd)
    os.fsync(directory)


def _publish(directory, name, value):
    """Publish complete bytes without replacing any prior result/version."""
    temporary = '.publish-' + secrets.token_hex(16)
    try:
        _write(directory, temporary, value)
        os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
    finally:
        if _exists(directory, temporary): os.unlink(temporary, dir_fd=directory)
        os.fsync(directory)


def _vectors(values, count):
    try:
        if type(values) is not list or len(values) != count: raise TemplateCacheError()
        for vector in values:
            if (type(vector) is not list or len(vector) != qwen.DIMENSIONS
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector)
                    or abs(math.hypot(*vector) - 1) > 1e-6):
                raise TemplateCacheError()
    except (ValueError, TypeError, OverflowError):
        raise TemplateCacheError() from None
    return values


def _sealed(value): return {**value, 'sha256': digest(value)}


def _unseal(value, fields):
    if type(value) is not dict or set(value) != fields | {'sha256'}: raise TemplateCacheError()
    content = {k: v for k, v in value.items() if k != 'sha256'}
    if value['sha256'] != digest(content): raise TemplateCacheError()
    return content


class TemplateCache:
    def __init__(self, base_url, directory=None):
        self.base_url = qwen.canonical_base_url(base_url)
        configured = directory if directory is not None else os.getenv('QWEN_INTENT_CACHE_DIRECTORY')
        try:
            self.path = Path(configured) if configured is not None else Path(__file__).resolve().parents[1] / 'data/intent-templates'
            if not self.path.is_absolute() or str(self.path) != str(self.path.resolve()): raise TemplateCacheError()
        except (OSError, ValueError, TypeError, RuntimeError):
            raise TemplateCacheError() from None

    def plan(self, snapshot):
        rows, texts = [], []
        if type(snapshot) is not dict or not snapshot: raise TemplateCacheError('embedding_template_limit')
        for key, group in snapshot.items():
            if type(group) not in (list, tuple) or not group: raise TemplateCacheError('embedding_template_limit')
            rows.append([str(key), list(group)])
            texts.extend(group)
        if not 1 <= len(texts) <= MAX_TEMPLATES: raise TemplateCacheError('embedding_template_limit')
        batches = [texts[i:i+qwen.BATCH_SIZE] for i in range(0, len(texts), qwen.BATCH_SIZE)]
        for batch in batches: qwen._payload(batch)  # validate entire plan before creating claims
        identity = {'version': VERSION, 'model': qwen.MODEL, 'dimensions': qwen.DIMENSIONS,
            'endpoint_sha256': hashlib.sha256(self.base_url.encode()).hexdigest(),
            'templates_sha256': digest(rows), 'template_count': len(texts)}
        return identity, batches

    @contextmanager
    def _version(self, identity, *, create=False):
        root, version = None, None
        try:
            if any(p.is_symlink() for p in (self.path, *self.path.parents)): raise TemplateCacheError()
            root = _directory(self.path, create=create)
            if create and not _exists(root, 'cache-root.json'):
                if os.listdir(root): raise TemplateCacheError()
                try: _write(root, 'cache-root.json', ROOT_MARKER)
                except FileExistsError: pass  # concurrent bootstrap can only validate
            if not _same(_read(root, 'cache-root.json', maximum=1024), ROOT_MARKER): raise TemplateCacheError()
            version = _directory(digest(identity), parent_fd=root, create=create)
            if create and not _exists(version, 'identity.json'):
                try: _write(version, 'identity.json', identity)
                except FileExistsError: pass
            if not _same(_read(version, 'identity.json', maximum=2048), identity): raise TemplateCacheError()
            yield version
        except (TemplateCacheError, qwen.EmbeddingError):
            raise
        except (OSError, ValueError, TypeError):
            raise TemplateCacheError() from None
        finally:
            if version is not None: os.close(version)
            if root is not None: os.close(root)

    def _load(self, version, identity):
        record = _unseal(_read(version, 'templates.json'), {'identity', 'vectors'})
        if not _same(record['identity'], identity): raise TemplateCacheError()
        return _vectors(record['vectors'], identity['template_count']), digest(record)

    def read(self, snapshot):
        identity, _ = self.plan(snapshot)
        with self._version(identity) as version:
            values, token = self._load(version, identity)
        groups, offset = {}, 0
        for key, texts in snapshot.items():
            groups[key] = values[offset:offset+len(texts)]
            offset += len(texts)
        return groups, token

    def status(self, snapshot):
        try:
            identity, batches = self.plan(snapshot)
            self.read(snapshot)
            return {'state': 'ready', 'version': digest(identity), 'templates': identity['template_count'], 'batches': len(batches)}
        except (TemplateCacheError, qwen.EmbeddingError) as error:
            return {'state': 'unprepared' if str(error) == 'embedding_templates_not_prepared' else 'invalid', 'error_code': str(error)}

    def import_prepared(self, snapshot, source, *, source_token, claim_digest, authorization_sha256):
        """Explicit operator recovery, zero sends; never erase a failed claim.

        Only a fully validated, separately prepared version may be published.
        A private immutable receipt binds the new authorization and source to
        the existing incomplete claim. No automatic recovery or model retry.
        """
        if (not isinstance(source, TemplateCache) or source.path == self.path
                or source.base_url != self.base_url
                or any(type(v) is not str or not re.fullmatch(r'[0-9a-f]{64}', v)
                       for v in (source_token, claim_digest, authorization_sha256))):
            raise TemplateCacheError()
        identity, batches = self.plan(snapshot)
        with source._version(identity) as source_fd:
            values, token = source._load(source_fd, identity)
        if token != source_token:
            raise TemplateCacheError()
        receipt = {'version': VERSION, 'action': 'explicit_import_after_incomplete_prepare',
            'identity': identity, 'source_token': token,
            'source_path_sha256': hashlib.sha256(str(source.path).encode()).hexdigest(),
            'incomplete_claim_digest': claim_digest, 'authorization_sha256': authorization_sha256}
        lock, root = None, None
        try:
            with self._version(identity) as target:
                if fcntl is None: raise TemplateCacheError()
                lock = os.open('prepare.lock', os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=target)
                _file(lock)
                try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError: raise TemplateCacheError('embedding_template_preparation_busy') from None
                allowed = {'identity.json', 'prepare.lock', 'templates.json'}
                allowed |= {f'batch-{i:03d}.{kind}.json' for i in range(len(batches)) for kind in ('claim', 'result')}
                if set(os.listdir(target)) - allowed: raise TemplateCacheError()
                incomplete = []
                for i, batch in enumerate(batches):
                    name = f'batch-{i:03d}'
                    # This deliberately narrow repair does not replace any
                    # previously validated/completed batch or merge versions.
                    if _exists(target, name + '.result.json'): raise TemplateCacheError()
                    if _exists(target, name + '.claim.json'):
                        claim = _read(target, name + '.claim.json', maximum=2048)
                        expected = {'version': VERSION, 'cache_version': digest(identity), 'batch': i,
                            'payload_sha256': hashlib.sha256(qwen._payload(batch)).hexdigest()}
                        if not _same(claim, expected): raise TemplateCacheError()
                        incomplete.append(digest(claim))
                if incomplete != [claim_digest]: raise TemplateCacheError()
                root = _directory(self.path)
                name = 'recovery-' + authorization_sha256 + '.json'
                if _exists(root, name):
                    if not _same(_read(root, name, maximum=4096), receipt): raise TemplateCacheError()
                else:
                    # A ready target with no matching receipt is not adopted.
                    if _exists(target, 'templates.json'): raise TemplateCacheError()
                    _write(root, name, receipt)
                if _exists(target, 'templates.json'):
                    if self._load(target, identity)[1] != token: raise TemplateCacheError()
                    imported = False
                else:
                    _publish(target, 'templates.json', _sealed({'identity': identity, 'vectors': values}))
                    imported = True
                return {'ready': True, 'templates': len(values), 'model_requests': 0,
                    'imported': imported, 'source_token': token, 'preserved_incomplete_claim': claim_digest}
        except TemplateCacheError:
            raise
        except (OSError, ValueError, TypeError):
            raise TemplateCacheError() from None
        finally:
            if root is not None: os.close(root)
            if lock is not None: os.close(lock)

    async def prepare(self, snapshot, fetch, *, max_requests):
        identity, batches = self.plan(snapshot)
        if type(max_requests) is not int or max_requests != len(batches):
            raise TemplateCacheError('embedding_template_request_limit')
        lock = None
        try:
            async with asyncio.timeout(PREPARE_DEADLINE):
                with self._version(identity, create=True) as version:
                    if fcntl is None: raise TemplateCacheError()
                    lock = os.open('prepare.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=version)
                    _file(lock)
                    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError: raise TemplateCacheError('embedding_template_preparation_busy') from None
                    allowed = {'identity.json', 'prepare.lock', 'templates.json'}
                    allowed |= {f'batch-{i:03d}.{kind}.json' for i in range(len(batches)) for kind in ('claim', 'result')}
                    if set(os.listdir(version)) - allowed: raise TemplateCacheError()
                    if _exists(version, 'templates.json'):
                        self._load(version, identity)
                        return {'ready': True, 'templates': identity['template_count'], 'batches': len(batches), 'new_requests': 0}
                    values, sent = [], 0
                    for i, batch in enumerate(batches):
                        claim_name, result_name = f'batch-{i:03d}.claim.json', f'batch-{i:03d}.result.json'
                        claim = {'version': VERSION, 'cache_version': digest(identity), 'batch': i,
                            'payload_sha256': hashlib.sha256(qwen._payload(batch)).hexdigest()}
                        if _exists(version, claim_name):
                            if not _same(_read(version, claim_name, maximum=2048), claim): raise TemplateCacheError()
                            if not _exists(version, result_name):
                                raise TemplateCacheError('embedding_template_preparation_incomplete')
                            result = _unseal(_read(version, result_name), {'claim', 'vectors'})
                            if not _same(result['claim'], claim): raise TemplateCacheError()
                            vectors = _vectors(result['vectors'], len(batch))
                        else:
                            if _exists(version, result_name): raise TemplateCacheError()
                            _write(version, claim_name, claim)  # nonrefundable conservative per-batch reservation
                            vectors = _vectors(await fetch(batch), len(batch))
                            if asyncio.current_task().cancelling(): raise asyncio.CancelledError()
                            _publish(version, result_name, _sealed({'claim': claim, 'vectors': vectors}))
                            sent += 1
                        values.extend(vectors)
                    _publish(version, 'templates.json', _sealed({'identity': identity, 'vectors': values}))
                    return {'ready': True, 'templates': len(values), 'batches': len(batches), 'new_requests': sent}
        except TimeoutError:
            raise TemplateCacheError('embedding_template_prepare_timeout') from None
        finally:
            if lock is not None: os.close(lock)
