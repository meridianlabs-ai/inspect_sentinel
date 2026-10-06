Guest-side stand-ins for the parts of `inspect_ai` that `inspect_sentinel`
imports beyond `inspect_ai.core`. `build.sh` vendors the real
`inspect_ai/core` (origin/main) and the real `inspect_sentinel` source next to
these. Every module here is a gap: something sentinel would have to stop
importing, or inspect_core would have to provide, for a guest to run it
without shims.
