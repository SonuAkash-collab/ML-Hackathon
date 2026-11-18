"""Serve the evaluator UI (run on Laptop 2).

    python ui/serve.py --api http://<laptop-1-ip>:8000 [--port 5500]

Opens the browser at the UI with the API address pre-filled.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import webbrowser
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="Serve the PathSnap UI")
    ap.add_argument("--api", default="http://localhost:8000", help="base URL of the PathSnap API on Laptop 1")
    ap.add_argument("--port", type=int, default=5500)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(HERE))
    url = f"http://localhost:{args.port}/index.html?api={quote(args.api, safe=':/')}"
    with socketserver.TCPServer(("0.0.0.0", args.port), handler) as httpd:
        print(f"PathSnap UI on {url}")
        print(f"API: {args.api}   (Ctrl+C to stop)")
        if not args.no_browser:
            webbrowser.open(url)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
