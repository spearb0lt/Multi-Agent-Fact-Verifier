"""Server entrypoint.

    python -m mas.main                      the API on port 8000
    python -m mas.main --reload             with autoreload, for development
    uvicorn mas.api.app:app                 the same thing, directly

The API and the CLI are two doors into the same system. Neither owns a run.
"""
from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mas.main", description="Run the Multi-Agentic API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    from .core import settings

    print(f"{settings.APP_NAME} API on http://{args.host}:{args.port}")
    print(f"  docs      http://localhost:{args.port}/docs")
    print(f"  config    http://localhost:{args.port}/api/config")

    uvicorn.run(
        "mas.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        # A run's trace is streamed, so a proxy that buffers would hold every
        # event until the run finished, which defeats the point of a live view.
        timeout_keep_alive=75,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
