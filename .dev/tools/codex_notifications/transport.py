"""Small read-only probe transport for the existing Codex Unix WebSocket.

No daemon start/resume/configuration calls are made by this module.
The caller supplies all RPC methods explicitly.
"""
import base64
import hashlib
import json
import os
import socket
import struct
import time
import threading
import queue


class RpcError(RuntimeError):
    def __init__(self, error):
        self.error = error
        super().__init__(str(error))


class RpcClient:
    def __init__(self, socket_path, timeout=10):
        self.socket = socket.socket(socket.AF_UNIX)
        self.socket.settimeout(timeout)
        self.socket.connect(socket_path)
        self.buffer = b""
        self.events = queue.Queue()
        self.next_id = 1
        self.timeout = timeout
        self.pending = {}
        self.pending_lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.closed = threading.Event()
        self.failure = None
        key = base64.b64encode(os.urandom(16)).decode()
        request = (
            "GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
            "Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
            f"Sec-WebSocket-Key: {key}\r\n\r\n"
        )
        self.socket.sendall(request.encode())
        while b"\r\n\r\n" not in self.buffer:
            chunk = self.socket.recv(4096)
            if not chunk:
                raise EOFError("Closed during HTTP upgrade")
            self.buffer += chunk
            if len(self.buffer) > 65536:
                raise ValueError("Oversized HTTP upgrade response")
        header, self.buffer = self.buffer.split(b"\r\n\r\n", 1)
        lines = header.decode("ascii").split("\r\n")
        if lines[0].split()[1] != "101":
            raise ValueError(f"WebSocket upgrade failed: {lines[0]}")
        headers = {k.lower(): v.strip() for k, v in (line.split(":", 1) for line in lines[1:] if ":" in line)}
        accept = base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
        ).digest()).decode()
        if headers.get("sec-websocket-accept", "").strip() != accept:
            raise ValueError("Invalid WebSocket accept")
        self.socket.settimeout(None)
        self.reader = threading.Thread(target=self._read_loop, daemon=True, name="codex-rpc-reader")
        self.reader.start()

    def close(self):
        self.closed.set()
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.socket.close()

    def _read(self, size):
        while len(self.buffer) < size:
            chunk = self.socket.recv(65536)
            if not chunk:
                raise EOFError("WebSocket closed")
            self.buffer += chunk
        data, self.buffer = self.buffer[:size], self.buffer[size:]
        return data

    def _send_frame(self, data, opcode=1):
        length = len(data)
        header = bytes([0x80 | opcode])
        if length < 126:
            header += bytes([0x80 | length])
        elif length < 65536:
            header += bytes([0xFE]) + struct.pack("!H", length)
        else:
            header += bytes([0xFF]) + struct.pack("!Q", length)
        mask = os.urandom(4)
        with self.send_lock:
            self.socket.sendall(header + mask + bytes(
                value ^ mask[index % 4] for index, value in enumerate(data)
            ))

    def send(self, obj):
        self._send_frame(json.dumps(obj, ensure_ascii=False).encode())

    def receive(self):
        parts = []
        while True:
            header = self._read(2)
            final, opcode = bool(header[0] & 0x80), header[0] & 0x0F
            length = header[1] & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._read(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read(8))[0]
            if length > 32 * 1024 * 1024:
                raise ValueError("Oversized WebSocket frame")
            mask = self._read(4) if header[1] & 0x80 else None
            data = self._read(length)
            if mask:
                data = bytes(value ^ mask[i % 4] for i, value in enumerate(data))
            if opcode == 8:
                raise EOFError("WebSocket close frame")
            if opcode == 9:
                self._send_frame(data, 10)
                continue
            if opcode == 10:
                continue
            if opcode not in (0, 1):
                raise ValueError(f"Unexpected WebSocket opcode: {opcode}")
            parts.append(data)
            if final:
                return json.loads(b"".join(parts))

    def _read_loop(self):
        try:
            while not self.closed.is_set():
                message = self.receive()
                if "id" in message and "method" not in message:
                    with self.pending_lock:
                        waiter = self.pending.get(message["id"])
                    if waiter is not None:
                        waiter.put(message)
                else:
                    self.events.put(message)
        except Exception as exc:
            self.failure = exc
            self.closed.set()
            with self.pending_lock:
                for waiter in self.pending.values():
                    waiter.put(exc)

    def call(self, method, params=None, timeout=None):
        if self.closed.is_set():
            raise ConnectionError("Codex RPC connection closed") from self.failure
        waiter = queue.Queue()
        with self.pending_lock:
            request_id = self.next_id
            self.next_id += 1
            self.pending[request_id] = waiter
        try:
            request = {"id": request_id, "method": method}
            if params is not None:
                request["params"] = params
            self.send(request)
            try:
                message = waiter.get(timeout=self.timeout if timeout is None else timeout)
            except queue.Empty:
                raise TimeoutError(f"RPC timed out: {method}") from None
            if isinstance(message, Exception):
                raise ConnectionError("Codex RPC connection failed") from message
            if "error" in message:
                raise RpcError(message["error"])
            return message["result"]
        finally:
            with self.pending_lock:
                self.pending.pop(request_id, None)

    def drain_events(self):
        messages = []
        while True:
            try:
                messages.append(self.events.get_nowait())
            except queue.Empty:
                return messages

    def initialize(self):
        result = self.call("initialize", {
            "clientInfo": {"name": "jevgo_notification_monitor", "version": "0.1.0"},
            "capabilities": {"experimentalApi": True},
        })
        self.send({"method": "initialized", "params": {}})
        return result
