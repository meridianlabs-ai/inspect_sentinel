#!/usr/bin/env bash
# Run each milestone against the artifacts build.sh produced.
set -euo pipefail
cd "$(dirname "$0")"
PY="${TOOLS:-build/tools}/venv/bin/python"
H=host_rs/target/release/sentinel-wasm-host
reports() { grep '^REPORT' | cut -d' ' -f3-; }
say() { printf '\n### %s\n' "$*"; }

say "M1: wasmtime-py host, blocking world, mocked generate"
"$PY" host_py/host.py build/m1_sync.wasm step.json --repeat 50

say "M2a: blocking world, 4 instances, async host (fibers), 500ms generate each"
$H sync build/m1_sync.wasm step.json --instances 4 --delay-ms 500

say "M2b: async world, one instance, 50 concurrent steps x 4-way guest fan-out"
$H async build/m1_async.wasm step.json --calls 50 --fanout 4 --delay-ms 500 | tail -1

say "M2c: asyncio/anyio features in the guest (after loop_patch)"
echo '{"asyncprobe": true}' > build/asyncprobe.json
FULL=1 $H async build/m1_async.wasm build/asyncprobe.json --delay-ms 50 | reports | python3 fmt.py asyncprobes

say "M2d: Python host, thread per instance"
"$PY" host_py/threads.py build/m1_sync.wasm step.json 4 | tail -1

say "M3: pydantic in the guest (valid, then invalid step)"
FULL=1 $H async build/m3_async.wasm step.json --delay-ms 10 | reports
python3 -c "
import json; d=json.load(open('step.json')); d['call']['arguments']=['x']; d['input'][0]['role']='robot'
json.dump(d, open('build/bad_step.json', 'w'))"
FULL=1 $H async build/m3_async.wasm build/bad_step.json | reports

say "M4: unchanged inspect_sentinel run_sentinel(concurrent([threshold({suspicion, trusted}), no_network(), no_destruction()]))"
for s in step_m4.json step_m4_network.json step_m4_rm.json; do
  echo "-- $s"
  FULL=1 $H async build/m4_async.wasm $s --delay-ms 200 | tee build/m4.out | grep wall
  reports < build/m4.out | python3 fmt.py m4
done

say "M5: sandbox probes (nothing granted), then with an env var and a dir granted"
"$PY" host_py/host.py build/m1_sync.wasm step.json --probe 2>/dev/null | python3 fmt.py probes
"$PY" host_py/host.py build/m1_sync.wasm step.json --probe --grant 2>/dev/null | python3 fmt.py probes 1 6

say "M6: measurements"
$H bench build/m1_async.wasm step.json --instances 2 --calls 10 --delay-ms 0 > build/bench.log; head -3 build/bench.log
$H bench build/m1_async.wasm.cwasm step.json --instances 50 --calls 500 --delay-ms 0
$H bench build/m4_async.wasm step_m4.json --instances 2 --calls 10 --delay-ms 0 > build/bench.log; head -3 build/bench.log
$H bench build/m4_async.wasm.cwasm step_m4.json --instances 50 --calls 500 --delay-ms 0
if command -v uv >/dev/null; then
  uv run -q --no-project --python 3.14 --with pydantic==2.13.5 --with anyio==4.11.0 --with shortuuid python native_bench.py
fi
for f in build/*.wasm build/*.cwasm; do
  printf '%-28s raw %6.1f MiB  gzip -9 %5.1f MiB  zstd -19 %5.1f MiB\n' "$f" \
    "$(echo "$(wc -c < "$f") / 1048576" | bc -l)" \
    "$(echo "$(gzip -9 -c "$f" | wc -c) / 1048576" | bc -l)" \
    "$(echo "$(zstd -19 -q -c "$f" | wc -c) / 1048576" | bc -l)"
done

say "Limits: epoch deadline, fuel, memory cap"
$H limits build/m1_async.wasm
