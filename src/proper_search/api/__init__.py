"""HTTP API. Requires the ``api`` extra: ``pip install proper-search[api]``."""

from .app import create_app

__all__ = ["create_app"]
