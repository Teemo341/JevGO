"""Protocol tests against a private fake Unix WebSocket, never Codex or Gmail."""
import base64
import hashlib
import json
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

from transport import RpcClient, RpcError


class FakeServer:
    def __init__(self, handler, valid_upgrade=True):
        self.handler = handler
        self.valid_upgrade = valid_upgrade
        self.errors = []
        self.frames = []
        self.done = threading.Event()
        self.temp = tempfile.TemporaryDirectory(prefix="jevgo-ws-test-")
        self.path = str(Path(self.temp.name) / "fake.sock")
        self.listener = socket.socket(socket.AF_UNIX)
        self.listener.bind(self.path)
        self.listener.listen(1)
        self.listener.settimeout(3)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            self.conn, _ = self.listener.accept()
            self.conn.settimeout(3)
            with self.conn:
                header = b""
                while not header.endswith(b"\r\n\r\n"):
                    header += self.exact(1)
                lines = header.decode().split("\r\n")
                assert lines[0] == "GET / HTTP/1.1"
                headers = {k.lower(): v.strip() for k, v in
                           (line.split(":", 1) for line in lines[1:] if ":" in line)}
                assert headers["upgrade"].lower() == "websocket"
                assert headers["connection"].lower() == "upgrade"
                assert headers["sec-websocket-version"] == "13"
                assert len(base64.b64decode(headers["sec-websocket-key"])) == 16
                assert "origin" not in headers
                accept = base64.b64encode(hashlib.sha1(
                    (headers["sec-websocket-key"] + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
                ).digest()).decode()
                if not self.valid_upgrade:
                    accept = "incorrect-accept"
                self.conn.sendall(("HTTP/1.1 101 Switching Protocols\r\n"
                                   "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                                   f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())
                self.handler(self)
        except BaseException as exc:
            self.errors.append(exc)
        finally:
            self.done.set()

    def exact(self, n):
        result = b""
        while len(result) < n:
            chunk = self.conn.recv(n - len(result))
            if not chunk:
                raise EOFError("Fake peer closed")
            result += chunk
        return result

    def read_frame(self):
        header = self.exact(2)
        assert header[0] & 0x80, "Client frame must be final"
        assert header[1] & 0x80, "Client frame must be masked"
        length = header[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", self.exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self.exact(8))[0]
        mask = self.exact(4)
        raw = self.exact(length)
        data = bytes(v ^ mask[i % 4] for i, v in enumerate(raw))
        self.frames.append((header[0] & 0x0F, data))
        return header[0] & 0x0F, data

    def read_json(self):
        opcode, data = self.read_frame()
        assert opcode == 1
        return json.loads(data)

    def send_frame(self, data, opcode=1, final=True):
        head = bytes([(0x80 if final else 0) | opcode])
        size = len(data)
        if size < 126:
            head += bytes([size])
        elif size < 65536:
            head += bytes([126]) + struct.pack("!H", size)
        else:
            head += bytes([127]) + struct.pack("!Q", size)
        self.conn.sendall(head + data)

    def send_json(self, obj):
        self.send_frame(json.dumps(obj, ensure_ascii=False).encode())

    def assert_finished(self, testcase):
        testcase.assertTrue(self.done.wait(4), "Fake server did not finish")
        if self.errors:
            raise self.errors[0]

    def close(self):
        self.listener.close()
        if hasattr(self, "conn"):
            try:
                self.conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.conn.close()
        self.thread.join(timeout=1)
        self.temp.cleanup()


class TransportTests(unittest.TestCase):
    def server(self, handler, **kwargs):
        server = FakeServer(handler, **kwargs)
        self.addCleanup(server.close)
        return server

    def client(self, server, **kwargs):
        client = RpcClient(server.path, **kwargs)
        self.addCleanup(client.close)
        return client

    def test_http_upgrade_initialize_and_masked_length_boundaries(self):
        values = ["测试", "x" * 200, "x" * 70000]

        def handler(s):
            first = s.read_json()
            self.assertEqual(first["method"], "initialize")
            self.assertTrue(first["params"]["capabilities"]["experimentalApi"])
            s.send_json({"id": first["id"], "result": {"version": "fake"}})
            self.assertEqual(s.read_json(), {"method": "initialized", "params": {}})
            for value in values:
                request = s.read_json()
                self.assertEqual(request["params"], {"text": value})
                s.send_json({"id": request["id"], "result": {"text": value}})

        server = self.server(handler)
        client = self.client(server)
        self.assertEqual(client.initialize(), {"version": "fake"})
        for value in values:
            self.assertEqual(client.call("fake/echo", {"text": value}), {"text": value})
        server.assert_finished(self)

    def test_fragmented_response_and_masked_pong(self):
        def handler(s):
            request = s.read_json()
            response = json.dumps({"id": request["id"], "result": {"ok": True}}).encode()
            cut = len(response) // 2
            s.send_frame(response[:cut], final=False)
            s.send_frame(b"keepalive", opcode=9)
            self.assertEqual(s.read_frame(), (10, b"keepalive"))
            s.send_frame(response[cut:], opcode=0)

        server = self.server(handler)
        client = self.client(server)
        self.assertEqual(client.call("fake/ping", {}), {"ok": True})
        server.assert_finished(self)

    def test_interleaved_server_request_is_queued_and_never_answered(self):
        def handler(s):
            request = s.read_json()
            # A server request may even reuse an outbound request's numeric ID.
            s.send_json({"id": request["id"], "method": "item/commandExecution/requestApproval",
                         "params": {"threadId": "fake-thread"}})
            s.send_json({"method": "thread/status/changed", "params": {
                "threadId": "fake-thread", "status": {"type": "active", "activeFlags": ["waitingOnApproval"]}}})
            s.send_json({"id": request["id"], "result": {"ok": True}})
            s.conn.settimeout(0.15)
            with self.assertRaises(socket.timeout):
                s.read_frame()

        server = self.server(handler)
        client = self.client(server)
        self.assertEqual(client.call("fake/read", {}), {"ok": True})
        events = client.drain_events()
        self.assertEqual([e["method"] for e in events], [
            "item/commandExecution/requestApproval", "thread/status/changed"])
        self.assertIn("id", events[0])
        server.assert_finished(self)
        self.assertEqual(len(server.frames), 1)

    def test_rpc_error_timeout_and_late_response_do_not_poison_next_call(self):
        def handler(s):
            request = s.read_json()
            s.send_json({"id": request["id"], "error": {"code": -32001, "message": "fake error"}})
            timed_out = s.read_json()
            time.sleep(0.1)
            s.send_json({"id": timed_out["id"], "result": {"late": True}})
            request = s.read_json()
            s.send_json({"id": request["id"], "result": {"next": True}})

        server = self.server(handler)
        client = self.client(server)
        with self.assertRaises(RpcError) as caught:
            client.call("fake/error", {})
        self.assertEqual(caught.exception.error["code"], -32001)
        with self.assertRaises(TimeoutError):
            client.call("fake/timeout", {}, timeout=0.03)
        self.assertEqual(client.call("fake/next", {}, timeout=1), {"next": True})
        self.assertEqual(client.pending, {})
        server.assert_finished(self)

    def test_bad_upgrade_accept_is_rejected(self):
        server = self.server(lambda s: None, valid_upgrade=False)
        raw = socket.socket(socket.AF_UNIX)
        self.addCleanup(raw.close)
        # Keep ownership of the raw socket so failed constructors are cleaned up.
        with patch("transport.socket", types.SimpleNamespace(AF_UNIX=socket.AF_UNIX, socket=lambda family: raw)):
            with self.assertRaisesRegex(ValueError, "Invalid WebSocket accept"):
                RpcClient(server.path)
        server.assert_finished(self)


if __name__ == "__main__":
    unittest.main()
