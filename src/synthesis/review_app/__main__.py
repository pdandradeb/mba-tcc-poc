"""CLI entrypoint for running the review web application."""

import sys
from .server import run_server

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port=port)
