"""Read model traffic that a proxy hands over into Inspect's types.

The first part of the sidecar described in `design/sidecar.md`: a request or a reply in a provider's own format, as it crossed the wire, read into the messages the model is sent or the output it gave. Each provider's SDK is an optional extra (`anthropic`, `openai`), imported the first time a body in its format is read.
"""

from ._wire import Format, UnreadableError, read_reply, read_request

__all__ = ["Format", "UnreadableError", "read_reply", "read_request"]
