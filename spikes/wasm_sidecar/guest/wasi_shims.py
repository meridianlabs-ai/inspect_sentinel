"""Import-time shims for the WASI CPython build (no `_ssl`).

anyio imports `ssl` at module level for its TLS streams; a guest never opens
a socket, so a stub module that satisfies the import is enough. Import this
before anything that imports anyio.
"""

import enum
import sys
import types

try:
    import ssl  # noqa: F401
except ModuleNotFoundError:
    ssl_stub = types.ModuleType("ssl")

    class SSLError(OSError):
        pass

    for _name in ("SSLEOFError", "SSLWantWriteError", "SSLWantReadError", "SSLSyscallError", "SSLZeroReturnError"):
        setattr(ssl_stub, _name, type(_name, (SSLError,), {}))
    ssl_stub.SSLError = SSLError
    for _name in ("SSLContext", "SSLObject", "SSLSocket", "MemoryBIO"):
        setattr(ssl_stub, _name, type(_name, (), {}))
    ssl_stub.Purpose = enum.Enum("Purpose", "SERVER_AUTH CLIENT_AUTH")
    ssl_stub.OP_IGNORE_UNEXPECTED_EOF = 0

    def create_default_context(*args: object, **kwargs: object) -> object:
        raise NotImplementedError("no TLS in the guest; use host.fetch")

    ssl_stub.create_default_context = create_default_context
    sys.modules["ssl"] = ssl_stub
