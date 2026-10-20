"""Generate the Python code for Envoy's ext_proc service, which the sidecar's Envoy shim serves.

Run with `make envoy-protos`. It downloads Envoy's API at `ENVOY_VERSION`, and the protos of cncf/xds and protoc-gen-validate at the versions that Envoy release pins, checking each download against the hash Envoy pins for it. It then generates the messages, their type stubs and the service's gRPC code into `PACKAGE`, with each import moved under it, so that the shim installs no top-level `envoy`, `udpa`, `xds` or `validate` package of its own.

The generated code requires a protobuf runtime at least as new as the protoc of `grpcio-tools` that generated it, and a grpcio at least as new as `grpcio-tools`, so the `envoy` extra's floors are those versions (the Makefile pins them).
"""

from __future__ import annotations

import ast
import hashlib
import io
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from google.protobuf import descriptor_pb2
from grpc_tools import protoc

ENVOY_VERSION = "1.39.4"
"""The Envoy release whose API the shim speaks."""

SERVICE = "envoy/service/ext_proc/v3/external_processor.proto"

REPOSITORY = Path(__file__).resolve().parent.parent
PACKAGE = "inspect_sentinel.sidecar.envoy._proto"
OUT = REPOSITORY / "src" / Path(*PACKAGE.split("."))

# the top-level packages the generated code imports, apart from protobuf's own
_ROOTS = ("envoy", "udpa", "xds", "validate")
_IMPORT = re.compile(rf"^from ({'|'.join(_ROOTS)})([. ])", re.MULTILINE)
_LEFT = re.compile(rf"^\s*(from|import) ({'|'.join(_ROOTS)})\b", re.MULTILINE)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        api = _envoy_api(work)
        pins = _pins(api / "bazel" / "repository_locations.bzl")
        xds = _fetch(pins["xds"], work / "xds")
        validate = _fetch(pins["com_envoyproxy_protoc_gen_validate"], work / "pgv")
        includes = [api, xds, validate, Path(protoc.__file__).parent / "_proto"]
        files = _closure(includes, work / "closure.pb")
        generated = work / "out"
        generated.mkdir()
        _protoc(includes, ["--python_out", "--mypy_out"], generated, files)
        _protoc(
            includes, ["--grpc_python_out", "--mypy_grpc_out"], generated, [SERVICE]
        )
        _install(generated, api.parent / "LICENSE")


def _envoy_api(work: Path) -> Path:
    url = (
        f"https://github.com/envoyproxy/envoy/archive/refs/tags/v{ENVOY_VERSION}.tar.gz"
    )
    with tarfile.open(fileobj=io.BytesIO(_download(url)), mode="r:gz") as archive:
        top = f"envoy-{ENVOY_VERSION}/"
        members = [
            member
            for member in archive.getmembers()
            if member.name.startswith(f"{top}api/") or member.name == f"{top}LICENSE"
        ]
        archive.extractall(work, members=members, filter="data")
    return work / f"envoy-{ENVOY_VERSION}" / "api"


def _pins(locations: Path) -> dict[str, dict[str, str]]:
    # Envoy's pins are a Starlark `dict(name = dict(...))`, read without
    # running it
    tree = ast.parse(locations.read_text())
    pins: dict[str, dict[str, str]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _named(node, "dict")):
            continue
        for keyword in node.keywords:
            value = keyword.value
            if keyword.arg is None or not (
                isinstance(value, ast.Call) and _named(value, "dict")
            ):
                continue
            fields: dict[str, str] = {}
            for field in value.keywords:
                if field.arg in ("version", "sha256", "strip_prefix") and isinstance(
                    field.value, ast.Constant
                ):
                    fields[field.arg] = str(field.value.value)
                elif field.arg == "urls" and isinstance(field.value, ast.List):
                    first = field.value.elts[0]
                    if isinstance(first, ast.Constant):
                        fields["url"] = str(first.value)
            pins[keyword.arg] = fields
    return pins


def _named(call: ast.Call, name: str) -> bool:
    return isinstance(call.func, ast.Name) and call.func.id == name


def _fetch(pin: dict[str, str], into: Path) -> Path:
    url = pin["url"].format(version=pin["version"])
    data = _download(url)
    digest = hashlib.sha256(data).hexdigest()
    if digest != pin["sha256"]:
        sys.exit(f"{url} has sha256 {digest}, not the {pin['sha256']} Envoy pins.")
    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            archive.extractall(into)
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            archive.extractall(into, filter="data")
    return into / pin.get("strip_prefix", "").format(version=pin["version"])


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def _closure(includes: list[Path], descriptors: Path) -> list[str]:
    # the service's proto and everything it imports, apart from protobuf's
    # own, which the protobuf runtime ships
    _run(
        [
            *_include(includes),
            "--include_imports",
            f"--descriptor_set_out={descriptors}",
            SERVICE,
        ]
    )
    found = descriptor_pb2.FileDescriptorSet()
    found.ParseFromString(descriptors.read_bytes())
    return [file.name for file in found.file if not file.name.startswith("google/")]


def _protoc(
    includes: list[Path], outputs: list[str], into: Path, files: list[str]
) -> None:
    _run([*_include(includes), *(f"{output}={into}" for output in outputs), *files])


def _include(includes: list[Path]) -> list[str]:
    return [f"-I{include}" for include in includes]


def _run(arguments: list[str]) -> None:
    status = protoc.main(["protoc", *arguments])
    if status != 0:
        sys.exit(f"protoc failed with status {status}.")


def _install(generated: Path, license: Path) -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for source in sorted(generated.rglob("*")):
        if source.is_dir():
            continue
        target = OUT / source.relative_to(generated)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text()
        moved = _IMPORT.sub(rf"from {PACKAGE}.\1\2", text)
        if _LEFT.search(moved):
            sys.exit(f"{source.name}: an import was left outside {PACKAGE}.")
        target.write_text(moved)
    for directory in [OUT, *(path for path in OUT.rglob("*") if path.is_dir())]:
        (directory / "__init__.py").touch()
    shutil.copyfile(license, OUT / "LICENSE")
    (OUT / "README.md").write_text(
        f"Generated by `make envoy-protos` from the protos of Envoy {ENVOY_VERSION}"
        " (envoyproxy/envoy, `api/`) and of the cncf/xds and"
        " bufbuild/protoc-gen-validate versions that release pins, all under the"
        " Apache License 2.0 in `LICENSE`. Don't edit these files: change"
        " `scripts/generate_envoy_protos.py` and generate them again.\n"
    )


if __name__ == "__main__":
    main()
