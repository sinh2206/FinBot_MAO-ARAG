from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the Streamlit UI.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="8501")
    parser.add_argument("--entrypoint", default="main.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        args.entrypoint,
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
    ]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
