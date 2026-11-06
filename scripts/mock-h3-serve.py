#!/usr/bin/env python3
"""A tiny stand-in for `h3 serve`, used to smoke-test the WebUI's optional
HTTP engine transport without CUDA, model weights or a GPU.

    python3 scripts/mock-h3-serve.py --port 18411

It implements GET /health, GET /progress, POST /generate and POST /unload with
the same wire format as the real C engine (see h3-spark.c/docs/PROTOCOL.md).
Every /generate echoes a fixed PNG byte string, so tests can assert that the
exact engine bytes reach the WebUI output gallery unchanged. It is a test
fixture only: it does NOT generate images and is never shipped in the NuGet
package.
"""
import argparse
import http.server
import json

PNG = b"\x89PNG\r\n\x1a\n" + b"QZI-MOCK-ENGINE-PNG"


class Handler(http.server.BaseHTTPRequestHandler):
    # One connection per request, exactly like the real h3 serve (which also
    # answers with `Connection: close`). Keeps the fixture deterministic.
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        print("GET " + self.path, flush=True)
        if self.path == "/health":
            self._json(200, {
                "ready": True, "service": "h3-serve", "resident_model": "qwen21",
                "busy": False, "queue_depth": 0, "served": 0, "rejected": 0,
                "uptime_s": 1.5,
                "vram": {"known": True, "free_mib": 8192, "total_mib": 24576},
            })
        elif self.path == "/progress":
            self._json(200, {"active": False})
        else:
            self._json(404, {"error": "not found", "status": 404})

    def do_POST(self):
        print("POST " + self.path, flush=True)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if self.path == "/generate":
            # Log the untouched wire bytes as well as the parsed form. CI greps
            # the WIRE line to assert the exact JSON H3ServeExecutor emits
            # (compact, literal 1.0, no forbidden fields).
            print("WIRE " + body.decode("utf-8", "replace"), flush=True)
            request = json.loads(body)
            print("GENERATE " + json.dumps(request, ensure_ascii=False), flush=True)
            self._send(200, PNG, "image/png")
        elif self.path == "/unload":
            self._json(200, {"released": {"dit": False}})
        else:
            self._json(404, {"error": "not found", "status": 404})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18411)
    args = parser.parse_args()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"mock h3 serve listening on http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
