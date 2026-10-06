//! Sidecar host spike: loads a componentize-py guest and serves its `host`
//! imports with async Rust functions (mocked `generate`).
//!
//! sentinel-wasm-host sync  <m1_sync.wasm>  <step.json> [--instances N] [--delay-ms D]
//! sentinel-wasm-host async <m1_async.wasm> <step.json> [--calls N] [--fanout F] [--delay-ms D]
//! sentinel-wasm-host bench <sync.wasm> <step.json> [--instances N] [--calls N]
//! sentinel-wasm-host limits <sync.wasm>

use wasmtime::{Result, bail, format_err};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use wasmtime::{
    Config, Engine, ResourceLimiter, Store,
    component::{Accessor, Component, HasSelf, Linker, ResourceTable},
};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};

mod sync_world {
    wasmtime::component::bindgen!({
        path: "../wit",
        world: "monitor-sync",
        imports: { default: async | trappable },
        exports: { default: async },
    });
}

mod async_world {
    wasmtime::component::bindgen!({
        path: "../wit",
        world: "monitor-async",
        imports: { default: async | trappable },
        exports: { default: async },
    });
}

static T0: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
fn ms() -> f64 {
    T0.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

/// Per-instance state. Credentials and endpoint config would live here (host side).
pub struct Ctx {
    wasi: WasiCtx,
    table: ResourceTable,
    id: usize,
    delay: Duration,
    limits: Limits,
    log: Arc<Mutex<Vec<String>>>,
    store: std::collections::HashMap<String, String>,
}

/// Tracks linear-memory high-water mark and enforces an optional cap.
#[derive(Default)]
pub struct Limits {
    pub peak_memory: usize,
    pub max_memory: Option<usize>,
}

impl ResourceLimiter for Limits {
    fn memory_growing(&mut self, _current: usize, desired: usize, _max: Option<usize>) -> Result<bool> {
        if let Some(cap) = self.max_memory {
            if desired > cap {
                return Ok(false);
            }
        }
        self.peak_memory = self.peak_memory.max(desired);
        Ok(true)
    }
    fn table_growing(&mut self, _c: usize, _d: usize, _m: Option<usize>) -> Result<bool> {
        Ok(true)
    }
}

impl WasiView for Ctx {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView { ctx: &mut self.wasi, table: &mut self.table }
    }
}

/// Canned `ModelOutput` JSON. A request with a `response_schema` (the real
/// `suspicion` monitor) gets a structured verdict; others get `SCORE: n` text.
fn mock_output(request: &str, tag: &str) -> String {
    let req: serde_json::Value = serde_json::from_str(request).unwrap_or_default();
    let input = req["input"].to_string();
    let structured = req["config"]["response_schema"].is_object();
    let content = if structured {
        let score = if input.contains("rm -rf") { 0.95 } else { 0.2 };
        serde_json::json!({"reasoning": format!("mock verdict ({tag})"), "score": score}).to_string()
    } else {
        format!("Reading disk usage is in scope ({tag}). SCORE: 2")
    };
    serde_json::json!({
        "model": "mockllm/model",
        "choices": [{
            "message": {"role": "assistant", "content": content, "source": "generate", "model": "mockllm/model"},
            "stop_reason": "stop"
        }],
        "usage": {"input_tokens": input.len() / 4, "output_tokens": 14, "total_tokens": input.len() / 4 + 14},
        "time": 0.0
    })
    .to_string()
}

fn known_endpoint(endpoint: &str) -> Result<String, String> {
    match endpoint {
        // A real host resolves the name to a URL and attaches a credential it holds.
        "allowlist" => Ok(r#"{"status": 200, "json": {"allowed": true}}"#.into()),
        other => Err(format!("unknown endpoint {other:?}")),
    }
}

// Blocking world: the import is synchronous to the guest. Because the host
// function is async, wasmtime runs the guest on a fiber and suspends it here.
impl sync_world::sentinel::spike::host::Host for Ctx {
    async fn generate(&mut self, request: String) -> wasmtime::Result<Result<String, String>> {
        let (id, delay) = (self.id, self.delay);
        self.log.lock().unwrap().push(format!("{:8.1}ms instance {id} generate start", ms()));
        tokio::time::sleep(delay).await; // stands in for the model HTTP call
        self.log.lock().unwrap().push(format!("{:8.1}ms instance {id} generate end", ms()));
        Ok(Ok(mock_output(&request, &format!("instance {id}"))))
    }
    async fn fetch(&mut self, endpoint: String, _request: String) -> wasmtime::Result<Result<String, String>> {
        Ok(known_endpoint(&endpoint))
    }
    async fn get(&mut self, key: String) -> wasmtime::Result<Option<String>> {
        Ok(self.store.get(&key).cloned())
    }
    async fn put(&mut self, key: String, value: String) -> wasmtime::Result<()> {
        self.store.insert(key, value);
        Ok(())
    }
}

// Async world: component-model async imports. Many may be in flight per instance.
static CALL_SEQ: AtomicUsize = AtomicUsize::new(0);

impl async_world::sentinel::spike::host_async::Host for Ctx {
    async fn get(&mut self, key: String) -> wasmtime::Result<Option<String>> {
        Ok(self.store.get(&key).cloned())
    }
    async fn put(&mut self, key: String, value: String) -> wasmtime::Result<()> {
        self.store.insert(key, value);
        Ok(())
    }
}

impl<T> async_world::sentinel::spike::host_async::HostWithStore<T> for HasSelf<Ctx> {
    async fn generate(accessor: &Accessor<T, Self>, request: String) -> wasmtime::Result<Result<String, String>> {
        let (id, delay, log) = accessor.with(|mut view| {
            let ctx = view.get();
            (ctx.id, ctx.delay, ctx.log.clone())
        });
        let n = CALL_SEQ.fetch_add(1, Ordering::SeqCst);
        log.lock().unwrap().push(format!("{:8.1}ms instance {id} generate #{n} start", ms()));
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        log.lock().unwrap().push(format!("{:8.1}ms instance {id} generate #{n} end", ms()));
        Ok(Ok(mock_output(&request, &format!("call {n}"))))
    }
    async fn fetch(_accessor: &Accessor<T, Self>, endpoint: String, _request: String) -> wasmtime::Result<Result<String, String>> {
        Ok(known_endpoint(&endpoint))
    }
    async fn sleep(_accessor: &Accessor<T, Self>, millis: u64) -> wasmtime::Result<()> {
        tokio::time::sleep(Duration::from_millis(millis)).await;
        Ok(())
    }
}

struct Opts {
    instances: usize,
    calls: usize,
    fanout: usize,
    delay: Duration,
}

fn parse_opts(args: &[String]) -> Result<Opts> {
    let mut o = Opts { instances: 1, calls: 1, fanout: 1, delay: Duration::from_millis(500) };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut v = || it.next().ok_or_else(|| format_err!("missing value for {a}"))?.parse::<u64>().map_err(wasmtime::Error::from);
        match a.as_str() {
            "--instances" => o.instances = v()? as usize,
            "--calls" => o.calls = v()? as usize,
            "--fanout" => o.fanout = v()? as usize,
            "--delay-ms" => o.delay = Duration::from_millis(v()?),
            other => bail!("unknown option {other}"),
        }
    }
    Ok(o)
}

fn engine(fuel: bool, epoch: bool) -> Result<Engine> {
    let mut config = Config::new();
    config.wasm_component_model(true);
    config.wasm_component_model_async(true);
    config.consume_fuel(fuel);
    config.epoch_interruption(epoch);
    Engine::new(&config)
}

fn new_store(engine: &Engine, id: usize, delay: Duration, log: Arc<Mutex<Vec<String>>>, max_memory: Option<usize>) -> Store<Ctx> {
    // Nothing inherited: no env, no preopened dirs, no network, no stdio.
    let mut wasi = WasiCtxBuilder::new();
    if std::env::var_os("GUEST_STDERR").is_some() {
        wasi.inherit_stderr(); // debugging only
    }
    let wasi = wasi.build();
    let mut store = Store::new(
        engine,
        Ctx {
            wasi,
            table: ResourceTable::new(),
            id,
            delay,
            limits: Limits { peak_memory: 0, max_memory },
            log,
            store: Default::default(),
        },
    );
    store.limiter(|ctx| &mut ctx.limits);
    store
}

fn load(engine: &Engine, path: &str) -> Result<Component> {
    let t = Instant::now();
    let component = Component::from_file(engine, path)?;
    eprintln!("compile {path}: {:.0}ms", t.elapsed().as_secs_f64() * 1000.0);
    Ok(component)
}

fn print_log(log: &Mutex<Vec<String>>) {
    for line in log.lock().unwrap().iter() {
        println!("  {line}");
    }
}

async fn run_sync(path: &str, step: String, o: Opts) -> Result<()> {
    let engine = engine(false, false)?;
    let component = load(&engine, path)?;
    let mut linker: Linker<Ctx> = Linker::new(&engine);
    wasmtime_wasi::p2::add_to_linker_async(&mut linker)?;
    sync_world::MonitorSync::add_to_linker::<_, HasSelf<_>>(&mut linker, |c| c)?;
    let pre = sync_world::MonitorSyncPre::new(linker.instantiate_pre(&component)?)?;
    let log = Arc::new(Mutex::new(Vec::new()));

    ms();
    let t = Instant::now();
    let tasks = (0..o.instances).map(|id| {
        let (pre, engine, step, log) = (pre.clone(), engine.clone(), step.clone(), log.clone());
        tokio::spawn(async move {
            let mut store = new_store(&engine, id, o.delay, log.clone(), None);
            let world = pre.instantiate_async(&mut store).await?;
            log.lock().unwrap().push(format!("{:8.1}ms instance {id} instantiated", ms()));
            let report = world.call_run_monitor(&mut store, &step).await?;
            log.lock().unwrap().push(format!("{:8.1}ms instance {id} report {:?}", ms(), report));
            Ok::<_, wasmtime::Error>(store.data().limits.peak_memory)
        })
    });
    let peaks: Vec<usize> = futures::future::try_join_all(tasks).await?.into_iter().collect::<Result<_>>()?;
    print_log(&log);
    println!(
        "sync world: {} instances, delay {:?} each: wall {:.0}ms; peak linear memory per instance {:.1} MiB",
        o.instances,
        o.delay,
        t.elapsed().as_secs_f64() * 1000.0,
        *peaks.iter().max().unwrap() as f64 / 1048576.0
    );
    Ok(())
}

async fn run_async(path: &str, step: String, o: Opts) -> Result<()> {
    let engine = engine(false, false)?;
    let component = load(&engine, path)?;
    let mut linker: Linker<Ctx> = Linker::new(&engine);
    wasmtime_wasi::p2::add_to_linker_async(&mut linker)?;
    async_world::MonitorAsync::add_to_linker::<_, HasSelf<_>>(&mut linker, |c| c)?;
    let pre = async_world::MonitorAsyncPre::new(linker.instantiate_pre(&component)?)?;
    let log = Arc::new(Mutex::new(Vec::new()));

    let mut store = new_store(&engine, 0, o.delay, log.clone(), None);
    let world = pre.instantiate_async(&mut store).await?;
    let mut step_json: serde_json::Value = serde_json::from_str(&step)?;
    step_json["fanout"] = o.fanout.into();
    let step = step_json.to_string();

    ms();
    let t = Instant::now();
    let reports = store
        .run_concurrent(async |accessor| {
            let calls = (0..o.calls).map(|_| world.call_run_monitor(accessor, step.clone()));
            futures::future::try_join_all(calls).await
        })
        .await??;
    print_log(&log);
    for (i, r) in reports.iter().enumerate() {
        match r {
            Ok(report) if std::env::var_os("FULL").is_some() => println!("REPORT {i} {report}"),
            Ok(report) => println!("  call {i}: ok {}", &report[..report.len().min(160)]),
            Err(e) => println!("  call {i}: err {e}"),
        }
    }
    println!(
        "async world: 1 instance, {} concurrent run-monitor calls x fanout {} = {} generates, delay {:?} each: wall {:.0}ms; peak linear memory {:.1} MiB",
        o.calls,
        o.fanout,
        o.calls * o.fanout,
        o.delay,
        t.elapsed().as_secs_f64() * 1000.0,
        store.data().limits.peak_memory as f64 / 1048576.0
    );
    Ok(())
}

fn rss_mib() -> f64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout).trim().parse::<f64>().unwrap_or(0.0) / 1024.0
}

fn stats(xs: &mut [f64]) -> String {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pick = |q: f64| xs[((xs.len() as f64 - 1.0) * q).round() as usize];
    format!("median {:.3}ms p90 {:.3}ms min {:.3}ms (n={})", pick(0.5), pick(0.9), xs[0], xs.len())
}

fn async_pre(engine: &Engine, component: &Component) -> Result<async_world::MonitorAsyncPre<Ctx>> {
    let mut linker: Linker<Ctx> = Linker::new(engine);
    wasmtime_wasi::p2::add_to_linker_async(&mut linker)?;
    async_world::MonitorAsync::add_to_linker::<_, HasSelf<_>>(&mut linker, |c| c)?;
    async_world::MonitorAsyncPre::new(linker.instantiate_pre(component)?)
}

async fn call_once(store: &mut Store<Ctx>, world: &async_world::MonitorAsync, step: &str) -> Result<Result<String, String>> {
    let step = step.to_string();
    store.run_concurrent(async |a| world.call_run_monitor(a, step).await).await?
}

/// Sizes, compile/AOT load, cold start, warm calls and memory for an async-world component.
async fn bench(path: &str, step: String, o: Opts) -> Result<()> {
    let log = Arc::new(Mutex::new(Vec::new()));
    let engine = engine(false, false)?;
    let rss_start = rss_mib();
    // A `.cwasm` is loaded as-is, so RSS is not inflated by compiling in-process.
    let (cwasm, compile_ms) = if path.ends_with(".cwasm") {
        (path.to_string(), None)
    } else {
        let t = Instant::now();
        let component = Component::from_file(&engine, path)?;
        let compile_ms = t.elapsed().as_secs_f64() * 1000.0;
        let cwasm = format!("{path}.cwasm");
        std::fs::write(&cwasm, component.serialize()?)?;
        (cwasm, Some(compile_ms))
    };
    let t = Instant::now();
    // SAFETY: produced by `serialize` with an engine of this configuration.
    let component = unsafe { Component::deserialize_file(&engine, &cwasm)? };
    let load_ms = t.elapsed().as_secs_f64() * 1000.0;
    let pre = async_pre(&engine, &component)?;
    println!("component {path}: {} bytes; AOT {cwasm}: {} bytes", std::fs::metadata(path)?.len(), std::fs::metadata(&cwasm)?.len());
    if let Some(ms) = compile_ms {
        println!("compile (Cranelift, then serialize) {ms:.0}ms");
    }
    println!("load precompiled {load_ms:.1}ms; host RSS after load {:.0} MiB (start {rss_start:.0})", rss_mib());

    let rss0 = rss_mib();
    let mut inst_ms = Vec::new();
    let mut instances = Vec::new();
    for id in 0..o.instances {
        let mut store = new_store(&engine, id, o.delay, log.clone(), None);
        let t = Instant::now();
        let world = pre.instantiate_async(&mut store).await?;
        inst_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        instances.push((store, world));
    }
    let rss1 = rss_mib();
    let mut first_ms = Vec::new();
    for (store, world) in instances.iter_mut() {
        let t = Instant::now();
        call_once(store, world, &step).await?.map_err(|e| format_err!("guest error {e}"))?;
        first_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let rss2 = rss_mib();
    let peak = instances.iter().map(|(s, _)| s.data().limits.peak_memory).max().unwrap_or(0);
    let (store, world) = &mut instances[0];
    let mut warm_ms = Vec::new();
    for _ in 0..o.calls {
        let t = Instant::now();
        call_once(store, world, &step).await?.map_err(|e| format_err!("guest error {e}"))?;
        warm_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let peak_after = store.data().limits.peak_memory;
    println!("instantiate: {}", stats(&mut inst_ms));
    println!("first call:  {}", stats(&mut first_ms));
    println!("warm call:   {}", stats(&mut warm_ms));
    println!(
        "memory: linear memory peak {:.1} MiB/instance after first call, {:.1} MiB after {} warm calls; host RSS +{:.1} MiB/instance at instantiation, +{:.1} MiB/instance after first call ({} instances)",
        peak as f64 / 1048576.0,
        peak_after as f64 / 1048576.0,
        o.calls,
        (rss1 - rss0) / o.instances as f64,
        (rss2 - rss0) / o.instances as f64,
        o.instances
    );
    if let Ok(meta) = call_once(store, world, "meta").await {
        println!("guest meta (pre-init check): {}", meta.unwrap_or_else(|e| e));
    }
    Ok(())
}

/// Epoch deadline, fuel and memory cap against a runaway or greedy guest (m1_async).
async fn limits(path: &str) -> Result<()> {
    let log = Arc::new(Mutex::new(Vec::new()));
    let engine = engine(true, true)?;
    let component = load(&engine, path)?;
    let pre = async_pre(&engine, &component)?;
    let ticker = engine.clone();
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_millis(10));
        ticker.increment_epoch();
    });
    let run = async |step: &str, deadline_ticks: u64, fuel: u64, max_memory: Option<usize>| -> Result<(String, u64, f64)> {
        let mut store = new_store(&engine, 0, Duration::from_millis(0), log.clone(), max_memory);
        store.set_fuel(u64::MAX)?;
        store.set_epoch_deadline(u64::MAX / 2);
        let world = pre.instantiate_async(&mut store).await?;
        store.set_fuel(fuel)?;
        store.epoch_deadline_trap();
        store.set_epoch_deadline(deadline_ticks);
        let t = Instant::now();
        let result = call_once(&mut store, &world, step).await;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let used = fuel - store.get_fuel().unwrap_or(0);
        let shown = match result {
            Ok(Ok(r)) => format!("ok {}", &r[..r.len().min(120)]),
            Ok(Err(e)) => format!("guest err {e}"),
            Err(e) => format!("trap: {}", e.root_cause()),
        };
        Ok((shown, used, ms))
    };
    let normal = std::fs::read_to_string("step.json")?;
    let (r, used, ms) = run(&normal, 1000, u64::MAX / 4, None).await?;
    println!("normal step:        {r} | fuel used {used} | {ms:.1}ms");
    let (r, _, ms) = run(r#"{"spin": true}"#, 50, u64::MAX / 4, None).await?;
    println!("spin, 500ms epoch:  {r} | {ms:.0}ms");
    let (r, used, ms) = run(r#"{"spin": true}"#, 1_000_000, 50_000_000, None).await?;
    println!("spin, 50M fuel:     {r} | fuel used {used} | {ms:.0}ms");
    let (r, _, _) = run(r#"{"alloc_mb": 16}"#, 1000, u64::MAX / 4, Some(64 << 20)).await?;
    println!("alloc 16MiB, cap 64MiB:  {r}");
    let (r, _, _) = run(r#"{"alloc_mb": 200}"#, 1000, u64::MAX / 4, Some(64 << 20)).await?;
    println!("alloc 200MiB, cap 64MiB: {r}");
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 3 && args[1] == "limits" {
        return limits(&args[2]).await;
    }
    if args.len() < 4 {
        bail!("usage: sentinel-wasm-host <sync|async|bench|limits> <component.wasm> <step.json> [options]");
    }
    let step = std::fs::read_to_string(&args[3])?;
    let o = parse_opts(&args[4..])?;
    match args[1].as_str() {
        "sync" => run_sync(&args[2], step, o).await,
        "async" => run_async(&args[2], step, o).await,
        "bench" => bench(&args[2], step, o).await,
        other => bail!("unknown command {other}"),
    }
}
