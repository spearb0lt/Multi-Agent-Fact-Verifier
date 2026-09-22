"""The HTTP surface. Thin by design: it creates runs and reads back the record.

The ASGI application deliberately is not re-exported here. Naming it `app` in
this package shadows the `mas.api.app` module with the FastAPI instance, so
`from mas.api import app` silently hands back an object where a module was
meant and every attribute on it raises. Everything that wants it already says
`mas.api.app:app`, which is unambiguous.
"""
from .app import create_app

__all__ = ["create_app"]
