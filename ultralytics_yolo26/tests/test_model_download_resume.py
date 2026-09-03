#!/usr/bin/env python3
"""Exercise automatic HTTP Range resume after a truncated response."""

from __future__ import annotations

import hashlib
import socket
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from scripts import model_setup

PAYLOAD = bytes(range(251)) * 12_000
REQUEST_RANGES: list[str | None] = []


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        range_header = self.headers.get("Range")
        REQUEST_RANGES.append(range_header)
        if len(REQUEST_RANGES) == 1:
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD[: 1_500_000])
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return

        offset = int(range_header.removeprefix("bytes=").removesuffix("-"))
        body = PAYLOAD[offset:]
        self.send_response(206)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Range", f"bytes {offset}-{len(PAYLOAD)-1}/{len(PAYLOAD)}")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original_chunk_size = model_setup.CHUNK_SIZE
    model_setup.CHUNK_SIZE = 1024 * 1024
    try:
        with tempfile.TemporaryDirectory() as directory:
            spec = model_setup.ModelSpec(
                "resume.bin",
                len(PAYLOAD),
                hashlib.sha256(PAYLOAD).hexdigest(),
                "test",
                url=f"http://127.0.0.1:{server.server_port}/resume.bin",
                verify_existing_sha256=True,
            )
            path = model_setup.download_model(
                spec, Path(directory), progress=False, max_attempts=3
            )
            assert path.read_bytes() == PAYLOAD
            assert REQUEST_RANGES[0] is None
            assert REQUEST_RANGES[1] == "bytes=1048576-", REQUEST_RANGES
            assert not Path(f"{path}.part").exists()
    finally:
        model_setup.CHUNK_SIZE = original_chunk_size
        server.shutdown()
        server.server_close()
    print({"requests": len(REQUEST_RANGES), "ranges": REQUEST_RANGES})
    print("MODEL_DOWNLOAD_AUTO_RESUME=PASS")


if __name__ == "__main__":
    main()
