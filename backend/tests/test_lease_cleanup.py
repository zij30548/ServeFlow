"""Cancellation must finish releasing the distributed conversation lease."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import pytest

from core.concurrency import redis_lease


def test_repeated_task_cancellation_waits_for_release():
    async def run():
        entered, releasing, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def release():
            releasing.set()
            await finish.wait()

        lock = SimpleNamespace(acquire=AsyncMock(return_value=True),
                               extend=AsyncMock(return_value=True),
                               release=AsyncMock(side_effect=release))
        client = SimpleNamespace(lock=lambda *args, **kwargs: lock)

        async def request():
            async with redis_lease(client, "owned"):
                entered.set()
                await asyncio.sleep(30)

        task = asyncio.create_task(request())
        await entered.wait()
        task.cancel()
        await releasing.wait()
        task.cancel()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert lock.release.await_count == 1
    asyncio.run(run())


def test_anyio_level_cancel_releases_before_scope_exits():
    async def run():
        released = asyncio.Event()

        async def release():
            await asyncio.sleep(0.01)
            released.set()

        lock = SimpleNamespace(acquire=AsyncMock(return_value=True),
                               extend=AsyncMock(return_value=True),
                               release=AsyncMock(side_effect=release))
        with anyio.CancelScope() as scope:
            async with redis_lease(SimpleNamespace(lock=lambda *a, **k: lock), "owned"):
                scope.cancel()
                await anyio.sleep(0)
        assert released.is_set()
        assert lock.release.await_count == 1
    asyncio.run(run())


def test_release_error_does_not_replace_original_error_or_log_payload(caplog):
    async def run():
        lock = SimpleNamespace(acquire=AsyncMock(return_value=True),
                               extend=AsyncMock(return_value=True),
                               release=AsyncMock(side_effect=RuntimeError("private-redis-value")))
        with pytest.raises(ValueError, match="business_failed"):
            async with redis_lease(SimpleNamespace(lock=lambda *a, **k: lock), "owned"):
                raise ValueError("business_failed")
        assert lock.release.await_count == 1
    asyncio.run(run())
    assert "private-redis-value" not in caplog.text
    assert "RuntimeError" in caplog.text
