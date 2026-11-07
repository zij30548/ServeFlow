"""Intent-only providers; one match always uses a single vector space."""
import asyncio
from contextlib import AsyncExitStack
import hashlib
import json
import math
import os
import time

import httpx

from core import protected_demo_quota as quota
from core import qwen_embedding as qwen
from core.intent_template_cache import TemplateCache, TemplateCacheError, digest

QWEN_MATCH_TIMEOUT = 17.0


class IntentEmbedding:
    MODEL = 'BAAI/bge-m3'
    URL = 'https://api.siliconflow.cn/v1/embeddings'

    def __init__(self):
        self.key = os.getenv('SILICONFLOW_API_KEY', '').strip()
        self.provider = os.getenv('INTENT_EMBEDDING_PROVIDER', 'auto').strip().lower()
        if self.provider not in {'auto', 'local', 'siliconflow', 'qwen'}:
            raise ValueError('INTENT_EMBEDDING_PROVIDER 只支持 auto/local/siliconflow/qwen')
        self.base_url = None
        if self.provider == 'qwen':
            self.key = qwen.validate_key(os.getenv('QWEN_EMBEDDING_API_KEY', ''))
            self.base_url = qwen.canonical_base_url(os.getenv('QWEN_EMBEDDING_BASE_URL', ''))
        elif quota.enabled() and self.provider != 'local':
            raise quota.ProtectedDemoQuotaError('demo_model_request_rejected')
        self.enabled = self.provider != 'local' and bool(self.key)
        self.batch_size = qwen.BATCH_SIZE if self.provider == 'qwen' else 32
        self.model = qwen.MODEL if self.provider == 'qwen' else self.MODEL
        self.source = 'qwen/' + qwen.MODEL if self.provider == 'qwen' else 'siliconflow/' + self.MODEL
        self.retry_at = 0.0
        self.last_error = None
        self.remote_templates = {}
        self.local_templates = {}
        self.lock = asyncio.Lock()
        self.template_cache = TemplateCache(self.base_url) if self.provider == 'qwen' else None
        self.last_diagnostic = {}
        self._last_cache_token = None

    def status(self):
        result = {'configured_provider': self.provider, 'online_configured': self.enabled,
                'model': self.model if self.enabled else 'local-ngram-256',
                'cooldown': time.monotonic() < self.retry_at, 'last_error': self.last_error}
        if self.provider == 'qwen':
            from core.intent_recognizer import _TEMPLATES
            result['template_cache'] = self.template_cache.status(_TEMPLATES)
            result['last_request'] = dict(self.last_diagnostic)
        return result

    async def cache_token(self, templates):
        """Make classification cache change when templates become ready/invalid."""
        snapshot = {k: tuple(v) for k, v in templates.items()}
        def token():
            try:
                identity, _ = self.template_cache.plan(snapshot)
            except (TemplateCacheError, qwen.EmbeddingError) as error:
                return error.code
            try: return self.template_cache.read(snapshot)[1]
            except TemplateCacheError as error:
                return error.code + ':' + digest(identity)
        try:
            async with asyncio.timeout(1):
                return await asyncio.to_thread(token)
        except TimeoutError:
            return 'embedding_template_cache_read_timeout'

    async def _qwen_vectors(self, message, snapshot, fingerprint, local_embed):
        try:
            async with asyncio.timeout(QWEN_MATCH_TIMEOUT):
                groups, token = await asyncio.to_thread(self.template_cache.read, snapshot)
                if token != self._last_cache_token:
                    self.retry_at = 0.
                    self._last_cache_token = token
                if time.monotonic() < self.retry_at:
                    return self._local_vectors(message, snapshot, fingerprint, local_embed)
                vector = (await qwen.request_embeddings([message], base_url=self.base_url, api_key=self.key,
                    purpose='query', diagnostic=self.last_diagnostic))[0]
                self.remote_templates = {fingerprint: groups}
                self.last_error = None
                return groups, vector, self.source
        except asyncio.CancelledError:
            raise
        except quota.ProtectedDemoQuotaError:
            raise
        except TemplateCacheError as error:
            self.remote_templates.clear()
            self.last_error = error.code
            self.last_diagnostic = {'purpose': 'query', 'phase': 'template_cache',
                'error_code': error.code, 'response_bytes': 0, 'http_status': None}
        except Exception as error:
            self.last_error = error.code if isinstance(error, qwen.EmbeddingError) else 'embedding_match_timeout' if isinstance(error, TimeoutError) else 'embedding_unavailable'
            self.retry_at = time.monotonic() + 60
        return self._local_vectors(message, snapshot, fingerprint, local_embed)

    def _local_vectors(self, message, snapshot, fingerprint, local_embed):
        if fingerprint not in self.local_templates:
            self.local_templates = {fingerprint: {key: [local_embed(t) for t in group] for key, group in snapshot.items()}}
        return self.local_templates[fingerprint], local_embed(message), 'local-ngram'

    async def vectors(self, message, templates, local_embed):
        # 快照防止 learn 在远程请求期间改变模板顺序或数量。
        snapshot = {key: tuple(texts) for key, texts in templates.items()}
        fingerprint = hashlib.sha256(json.dumps([(str(k), v) for k, v in snapshot.items()], ensure_ascii=True).encode()).hexdigest()
        if self.provider == 'qwen':
            return await self._qwen_vectors(message, snapshot, fingerprint, local_embed)
        if self.enabled and time.monotonic() >= self.retry_at:
            try:
                async with asyncio.timeout(8):
                    async with AsyncExitStack() as stack:
                        client = None
                        if self.provider != 'qwen':
                            client = await stack.enter_async_context(httpx.AsyncClient(timeout=6, follow_redirects=False))
                        async with self.lock:
                            if time.monotonic() < self.retry_at:
                                raise RuntimeError('cooldown')
                            if fingerprint not in self.remote_templates:
                                texts = [text for group in snapshot.values() for text in group]
                                vectors = []
                                for start in range(0, len(texts), self.batch_size):
                                    vectors.extend(await self._request(client, texts[start:start+self.batch_size]))
                                if len({len(v) for v in vectors}) != 1:
                                    raise ValueError('template_dimension_mismatch')
                                groups, offset = {}, 0
                                for key, group in snapshot.items():
                                    groups[key] = vectors[offset:offset+len(group)]
                                    offset += len(group)
                                self.remote_templates = {fingerprint: groups}
                            groups = self.remote_templates[fingerprint]
                        query = (await self._request(client, [message]))[0]
                        if any(len(v) != len(query) for group in groups.values() for v in group):
                            self.remote_templates.clear()
                            raise ValueError('query_dimension_mismatch')
                        self.last_error = None
                        return groups, query, self.source
            except asyncio.CancelledError:
                raise
            except quota.ProtectedDemoQuotaError:
                # Broken/exhausted persistent accounting must never be hidden
                # by local fallback. The HTTP request scope also stays blocked.
                raise
            except Exception as ex:
                # 不记录异常全文，避免暴露请求内容或认证信息。
                self.last_error = (ex.code if isinstance(ex, qwen.EmbeddingError) else
                                   f'http_{ex.response.status_code}' if isinstance(ex, httpx.HTTPStatusError) else type(ex).__name__)
                self.retry_at = time.monotonic() + 60
        return self._local_vectors(message, snapshot, fingerprint, local_embed)

    async def _request(self, client, texts):
        if self.provider == 'qwen':
            return await qwen.request_embeddings(texts, base_url=self.base_url, api_key=self.key)
        response = await client.post(self.URL, headers={'Authorization': 'Bearer ' + self.key},
                                     json={'model': self.MODEL, 'input': texts, 'encoding_format': 'float'})
        response.raise_for_status()
        rows = response.json().get('data')
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise ValueError('invalid_embedding_count')
        result = [None] * len(texts)
        dimension = None
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('invalid_embedding_row')
            index, vector = row.get('index'), row.get('embedding')
            if type(index) is not int or not 0 <= index < len(texts) or result[index] is not None:
                raise ValueError('invalid_embedding_index')
            if not isinstance(vector, list) or not vector or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector):
                raise ValueError('invalid_embedding_vector')
            dimension = dimension or len(vector)
            norm = math.sqrt(sum(x*x for x in vector))
            if len(vector) != dimension or not math.isfinite(norm) or norm == 0:
                raise ValueError('invalid_embedding_dimension_or_norm')
            result[index] = [x/norm for x in vector]
        return result
