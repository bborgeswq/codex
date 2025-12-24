"""
AI Crypto Trader package initialization.

Exposes key configuration so downstream modules can rely on a single source
of truth for environment-controlled settings.
"""

from .config import Config  # re-export for convenience

__all__ = ["Config"]
