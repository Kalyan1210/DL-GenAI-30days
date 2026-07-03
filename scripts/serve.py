#!/usr/bin/env python3
"""Serve the DL-GenAI portal and all browser interactives from the repo root."""

from __future__ import annotations

import argparse
import http.server
import socket
import socketserver
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):
        if args and "404" in str(args[1]):
            return
        super().log_message(fmt, *args)


def pick_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("", preferred))
            return preferred
        except OSError:
            s.bind(("", 0))
            return s.getsockname()[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--port", type=int, default=8080, help="port (default: 8080)")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()

    port = pick_port(args.port)
    url = f"http://localhost:{port}/"

    print()
    print("  DL-GenAI 30 Days — local server")
    print("  ───────────────────────────────")
    print(f"  Home:                  {url}")
    print(f"  Interactives hub:      {url}interactives/")
    print(f"  Inference Engineering: {url}interactives/inference-engineering/")
    print()
    print("  Press Ctrl+C to stop.")
    print()

    if not args.no_open:
        webbrowser.open(url)

    with socketserver.TCPServer(("", port), QuietHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
