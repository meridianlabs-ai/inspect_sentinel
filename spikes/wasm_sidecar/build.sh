#!/usr/bin/env bash
# Build every component and both hosts for the WASM sidecar spike.
#
# Everything lands under build/ (gitignored); tools go under $TOOLS
# (default build/tools), never into the system. Each step is skipped when its
# output exists, so re-running is cheap. Tested on macOS arm64; the wasi-sdk
# download is the only platform-specific step (set WASI_SDK_ASSET for another).
#
#   ./build.sh            # build everything
#   TOOLS=/some/dir ./build.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BUILD="$HERE/build"
TOOLS="${TOOLS:-$BUILD/tools}"
mkdir -p "$BUILD" "$TOOLS"

# --- pins ---------------------------------------------------------------------
COMPONENTIZE_PY=0.25.1          # embeds CPython 3.14 (dicej/cpython v3.14.0-wasi-sdk-30); async/WASIp3 since 0.19
WASMTIME_PY=49.0.0              # wasmtime-py: components yes, async host functions no
RUST_TOOLCHAIN=1.99.0           # host build and the pydantic-core cross build
# wasmtime crate =46.0.1 is pinned in host_rs/Cargo.toml (what componentize-py 0.25.1 tests against)
WASI_SDK_VERSION=30             # must match the wasi-sdk componentize-py's CPython was built with
WASI_SDK_ASSET="${WASI_SDK_ASSET:-wasi-sdk-30.0-arm64-macos.tar.gz}"
WASI_SDK_SHA256="${WASI_SDK_SHA256:-2c2ed99296857e60fd14c3f40fe226231f296409502491094704089c31a16740}"
PYDANTIC=2.13.5
PYDANTIC_CORE=2.46.5            # pydantic 2.13.5 pins it
PYDANTIC_CORE_SDIST=https://files.pythonhosted.org/packages/af/f9/8a06bea35ef8daf588f707784c973a7046e0034c8d8cfb08828eeffb8b75/pydantic_core-2.46.5.tar.gz
PYDANTIC_CORE_SHA256=10416c15b8839ecc4ef4d0885da76da6fd0f67333a0eb8aff6d93c4b8f2910fc
PYYAML_SDIST=https://files.pythonhosted.org/packages/05/8e/961c0007c59b8dd7729d542c61a4d537767a59645b82a0b521206e1e25c2/pyyaml-6.0.3.tar.gz
PYYAML_SHA256=d76623373421df22fb4cf8817020cbb7ef15c725b9d5e45f17e189bfc384190f
INSPECT_AI_REPO=https://github.com/UKGovernmentBEIS/inspect_ai.git
INSPECT_AI_REF="${INSPECT_AI_REF:-8379140fd6d2184b669b6f3cea58e18009bde507}"  # origin/main, 2026-10-06
PURE_DEPS=(anyio==4.11.0 sniffio==1.3.1 idna==3.20 typing_extensions==4.16.0 shortuuid==1.0.13)
PYDANTIC_DEPS=(pydantic==$PYDANTIC annotated-types==0.7.0 typing-inspection==0.4.2 typing_extensions==4.16.0)

step() { printf '\n== %s\n' "$*"; }
fetch() {  # url sha256 dest
  [ -f "$3" ] || curl -sSL "$1" -o "$3"
  echo "$2  $3" | shasum -a 256 -c -
}

# --- tools ----------------------------------------------------------------------
step "Python tools (componentize-py $COMPONENTIZE_PY, wasmtime-py $WASMTIME_PY)"
if [ ! -x "$TOOLS/venv/bin/componentize-py" ]; then
  uv venv -q --python 3.12 "$TOOLS/venv"
  VIRTUAL_ENV="$TOOLS/venv" uv pip install -q "componentize-py==$COMPONENTIZE_PY" "wasmtime==$WASMTIME_PY"
fi
CPY="$TOOLS/venv/bin/componentize-py"
"$CPY" --version

step "Rust $RUST_TOOLCHAIN (+ wasm32-wasip1)"
export RUSTUP_HOME="$TOOLS/rustup" CARGO_HOME="$TOOLS/cargo"
if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
  curl -sSf https://sh.rustup.rs -o "$TOOLS/rustup-init.sh"
  sh "$TOOLS/rustup-init.sh" -y -q --no-modify-path --profile minimal --default-toolchain "$RUST_TOOLCHAIN"
fi
export PATH="$CARGO_HOME/bin:$PATH"
rustup target add --toolchain "$RUST_TOOLCHAIN" wasm32-wasip1 >/dev/null
cargo --version

step "wasi-sdk $WASI_SDK_VERSION"
fetch "https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-$WASI_SDK_VERSION/$WASI_SDK_ASSET" "$WASI_SDK_SHA256" "$TOOLS/$WASI_SDK_ASSET"
WASI_SDK="$TOOLS/${WASI_SDK_ASSET%.tar.gz}"
[ -d "$WASI_SDK" ] || tar xzf "$TOOLS/$WASI_SDK_ASSET" -C "$TOOLS"

# --- guest dependencies -------------------------------------------------------
step "pure-Python guest deps"
[ -d "$BUILD/site-pure/anyio" ] || uv pip install -q --target "$BUILD/site-pure" --python-version 3.14 \
  --python-platform linux --only-binary :all: --no-deps "${PURE_DEPS[@]}"

step "pydantic-core $PYDANTIC_CORE for wasm32-wasi, CPython 3.14 (no published wheel)"
# PyPI has only an Emscripten wasm wheel; benbrandt/wasi-wheels stops at
# cp313 and componentize-py >= 0.18 embeds 3.14, so cross-compile it here as a
# PIC shared library that componentize-py links dynamically (the recipe is
# benbrandt/wasi-wheels' src/build/pydantic.rs, with PyO3 told the target
# Python by config file instead of a CPython WASI build).
SO="$BUILD/site-pyd/pydantic_core/_pydantic_core.cpython-314-wasm32-wasi.so"
if [ ! -f "$SO" ]; then
  fetch "$PYDANTIC_CORE_SDIST" "$PYDANTIC_CORE_SHA256" "$TOOLS/pydantic_core-$PYDANTIC_CORE.tar.gz"
  [ -d "$TOOLS/pydantic_core-$PYDANTIC_CORE" ] || tar xzf "$TOOLS/pydantic_core-$PYDANTIC_CORE.tar.gz" -C "$TOOLS"
  cat > "$TOOLS/pyo3-wasi-ext.txt" <<'EOF'
implementation=CPython
version=3.14
shared=true
abi3=false
pointer_width=32
build_flags=
suppress_build_script_link_lines=true
EOF
  SYSROOT="$WASI_SDK/share/wasi-sysroot"
  (cd "$TOOLS/pydantic_core-$PYDANTIC_CORE" && \
    PYO3_CONFIG_FILE="$TOOLS/pyo3-wasi-ext.txt" \
    RUSTFLAGS="-C link-args=-L$SYSROOT/lib/wasm32-wasip1/ -C link-self-contained=no -C link-args=--experimental-pic -C link-args=--shared -C link-args=--unresolved-symbols=import-dynamic -C relocation-model=pic -C linker-plugin-lto=yes -C opt-level=s -C lto=true -C codegen-units=1" \
    cargo "+$RUST_TOOLCHAIN" rustc --release --target wasm32-wasip1 --lib --features pyo3/extension-module --crate-type cdylib)
  uv pip install -q --target "$BUILD/site-pyd" --python-version 3.14 --python-platform linux \
    --only-binary :all: --no-deps "${PYDANTIC_DEPS[@]}"
  cp -R "$TOOLS/pydantic_core-$PYDANTIC_CORE/python/pydantic_core" "$BUILD/site-pyd/"
  cp "$TOOLS/pydantic_core-$PYDANTIC_CORE/target/wasm32-wasip1/release/_pydantic_core.wasm" "$SO"
fi
ls -la "$SO"

step "M4 import path: shims + inspect_ai.core ($INSPECT_AI_REF) + inspect_sentinel (this checkout) + PyYAML"
if [ ! -d "$BUILD/m4-path/inspect_ai/core" ]; then
  rm -rf "$BUILD/m4-path" "$TOOLS/inspect_ai"
  git clone -q --filter=blob:none --no-checkout "$INSPECT_AI_REPO" "$TOOLS/inspect_ai"
  git -C "$TOOLS/inspect_ai" archive "$INSPECT_AI_REF" src/inspect_ai/core | tar -x -C "$TOOLS/inspect_ai"
  fetch "$PYYAML_SDIST" "$PYYAML_SHA256" "$TOOLS/pyyaml-6.0.3.tar.gz"
  tar xzf "$TOOLS/pyyaml-6.0.3.tar.gz" -C "$TOOLS"
  mkdir -p "$BUILD/m4-path"
  cp -R "$HERE/guest/inspect_shim/inspect_ai" "$BUILD/m4-path/"
  cp -R "$TOOLS/inspect_ai/src/inspect_ai/core" "$BUILD/m4-path/inspect_ai/"
  cp -R "$TOOLS/pyyaml-6.0.3/lib/yaml" "$BUILD/m4-path/"
fi
rm -rf "$BUILD/m4-path/inspect_sentinel"
cp -R "$REPO/src/inspect_sentinel" "$BUILD/m4-path/"

# --- components ---------------------------------------------------------------
cd "$HERE"
build() {  # world app output paths...
  local world=$1 app=$2 out=$3; shift 3
  local args=(-p guest)
  for p in "$@"; do args+=(-p "$p"); done
  step "component $out ($world, $app)"
  /usr/bin/time -p "$CPY" -q -d wit -w "$world" componentize "$app" "${args[@]}" -o "build/$out" 2>&1 | grep -E '^real' || true
  ls -la "build/$out"
}
build monitor-sync  app_m1_sync  m1_sync.wasm
build monitor-async app_m1_async m1_async.wasm build/site-pure
build monitor-async app_m3_async m3_async.wasm build/site-pure build/site-pyd
build monitor-async app_m4_async m4_async.wasm build/m4-path build/site-pure build/site-pyd

# --- host -----------------------------------------------------------------------
step "Rust host (wasmtime 46.0.1, component-model async)"
(cd host_rs && cargo "+$RUST_TOOLCHAIN" build --release -q)
ls -la host_rs/target/release/sentinel-wasm-host
echo
echo "done; see demo.sh"
