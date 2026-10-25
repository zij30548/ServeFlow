"""Explicitly initialized synthetic customer/order data, separate from RAG."""

from .store import DemoBusinessStore, DemoStoreError

__all__ = ["DemoBusinessStore", "DemoStoreError"]
