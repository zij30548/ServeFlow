"""Explicit operator commands for Qwen templates. Importing never sends."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from core import qwen_embedding as qwen
from core import protected_demo_quota as quota
from core.intent_template_cache import TemplateCache, TemplateCacheError


def configured_cache():
    if os.getenv('INTENT_EMBEDDING_PROVIDER') != 'qwen':
        raise qwen.EmbeddingError('embedding_provider_required')
    return TemplateCache(os.getenv('QWEN_EMBEDDING_BASE_URL', ''))


def template_status():
    from core.intent_recognizer import _TEMPLATES
    return configured_cache().status(_TEMPLATES)


async def prepare_templates(*, max_requests, diagnostics=None):
    from core.intent_recognizer import _TEMPLATES
    cache = configured_cache()
    key = qwen.validate_key(os.getenv('QWEN_EMBEDDING_API_KEY', ''))
    snapshot = {k: tuple(v) for k, v in _TEMPLATES.items()}
    # This command never initializes or resets the protected global ledger.
    if quota.enabled(): quota.verify()
    async def fetch(texts):
        record = {}
        try:
            return await qwen.request_embeddings(texts, base_url=cache.base_url, api_key=key,
                                                  purpose='prepare', diagnostic=record)
        finally:
            if diagnostics is not None: diagnostics.append(record)
    return await cache.prepare(snapshot, fetch, max_requests=max_requests)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Read status or explicitly prepare private intent templates; preparation may incur model charges.')
    parser.add_argument('command', choices=('status', 'prepare'))
    parser.add_argument('--max-requests', type=int)
    args = parser.parse_args(argv)
    if (args.command == 'prepare') != (args.max_requests is not None):
        parser.error('prepare requires --max-requests; status does not accept it')
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False)
    diagnostics = []
    try:
        result = (template_status() if args.command == 'status' else
                  asyncio.run(prepare_templates(max_requests=args.max_requests, diagnostics=diagnostics)))
        print(json.dumps({'result': result, 'request_diagnostics': diagnostics}, ensure_ascii=True))
        return 0
    except (TemplateCacheError, qwen.EmbeddingError, quota.ProtectedDemoQuotaError) as error:
        code = error.code
    except KeyboardInterrupt:
        code = 'embedding_cancelled'
    except Exception:
        code = 'embedding_template_cache_invalid'
    print(json.dumps({'result': 'stopped', 'error_code': code, 'request_diagnostics': diagnostics}, ensure_ascii=True))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
