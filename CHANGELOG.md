# Changelog

## 0.1.0 (2026-10-10)


### Features

* [@monitor](https://github.com/monitor) and [@protocol](https://github.com/protocol) decorators ([#3](https://github.com/meridianlabs-ai/inspect_sentinel/issues/3)) ([1f4ea44](https://github.com/meridianlabs-ai/inspect_sentinel/commit/1f4ea4402170c03c6e6b9d2bf35acb4ee6c662c3))
* build a sentinel from configuration and record it back ([#10](https://github.com/meridianlabs-ai/inspect_sentinel/issues/10)) ([a0e8562](https://github.com/meridianlabs-ai/inspect_sentinel/commit/a0e8562293cd307fc02f4d08f7142a5722e277bb))
* Context, RunnerContext, Host and Recorder interfaces ([84737b0](https://github.com/meridianlabs-ai/inspect_sentinel/commit/84737b0ea10fc7fef59176d7130276917d36cfda))
* each monitor or protocol function handles one stage ([#29](https://github.com/meridianlabs-ai/inspect_sentinel/issues/29)) ([76d78ce](https://github.com/meridianlabs-ai/inspect_sentinel/commit/76d78cee5948b5e3d1d54f849f548f31d86ab867))
* explicit MonitorGroup/ProtocolGroup, decorator name and version, and a guard on direct calls ([#20](https://github.com/meridianlabs-ai/inspect_sentinel/issues/20)) ([853fe5d](https://github.com/meridianlabs-ai/inspect_sentinel/commit/853fe5df0440eb89b6b8b23a3693bf21125003a5))
* export Reported and Report; keyword-only Context ([85eaa94](https://github.com/meridianlabs-ai/inspect_sentinel/commit/85eaa944caa23f437d677306c7d5355e380db70d))
* export the author-facing types ([43cc232](https://github.com/meridianlabs-ai/inspect_sentinel/commit/43cc232b644690f0de8d3ddd79b9665a784092e7))
* get_model-style model and role on the host, Context.input_text, and quieter concurrent explanations ([#30](https://github.com/meridianlabs-ai/inspect_sentinel/issues/30)) ([8d466d2](https://github.com/meridianlabs-ai/inspect_sentinel/commit/8d466d2b9648cf3d09fcb0dca540699206611623))
* helpers for rules that check tool calls ([#49](https://github.com/meridianlabs-ai/inspect_sentinel/issues/49)) ([ddb337a](https://github.com/meridianlabs-ai/inspect_sentinel/commit/ddb337a7e510e38844567d248d9bccaf867da3dd))
* human(), a person who decides what a chain escalates ([#34](https://github.com/meridianlabs-ai/inspect_sentinel/issues/34)) ([716cf90](https://github.com/meridianlabs-ai/inspect_sentinel/commit/716cf90a7d10c6eea8195de07a03be5a865263a2))
* monitor and protocol data types ([0301631](https://github.com/meridianlabs-ai/inspect_sentinel/commit/0301631ea9060922f1563f7ef4481dee2492c644))
* monitors report failures and the protocols reading them decide ([#41](https://github.com/meridianlabs-ai/inspect_sentinel/issues/41)) ([00a482d](https://github.com/meridianlabs-ai/inspect_sentinel/commit/00a482dde33402fc90a3475a0e3a2c0e28d07a93))
* multi-function monitor and protocol factories, one runner per family ([#8](https://github.com/meridianlabs-ai/inspect_sentinel/issues/8)) ([fabae2e](https://github.com/meridianlabs-ai/inspect_sentinel/commit/fabae2e7e8ac6eddb2065f3f8f93944f681d7b3a))
* observe, concurrent and threshold protocols, final decisions, and resolve_sentinel ([#7](https://github.com/meridianlabs-ai/inspect_sentinel/issues/7)) ([33915f1](https://github.com/meridianlabs-ai/inspect_sentinel/commit/33915f1fec2c133e2b45584f1e5fd75f1aefe7eb))
* portable monitors and protocols are checked when configured ([#57](https://github.com/meridianlabs-ai/inspect_sentinel/issues/57)) ([cb177bf](https://github.com/meridianlabs-ai/inspect_sentinel/commit/cb177bfd6b2c7cec8a3772c634494e56ff984dd5))
* Protocol, Decision.proceed() and a named run_children result ([#19](https://github.com/meridianlabs-ai/inspect_sentinel/issues/19)) ([fbc5893](https://github.com/meridianlabs-ai/inspect_sentinel/commit/fbc58933e5ddcd5572ed3fb60edc16588c4eb91a))
* record each sentinel's version in the log ([#36](https://github.com/meridianlabs-ai/inspect_sentinel/issues/36)) ([a766965](https://github.com/meridianlabs-ai/inspect_sentinel/commit/a766965ff5ed9b2e9f03044c461de0a143a5b539))
* record which factory produced each report ([#11](https://github.com/meridianlabs-ai/inspect_sentinel/issues/11)) ([fc8a846](https://github.com/meridianlabs-ai/inspect_sentinel/commit/fc8a846956ffe84c9753eb2c9443a623baa337e7))
* references on reports and per-dimension thresholds ([#22](https://github.com/meridianlabs-ai/inspect_sentinel/issues/22)) ([9c8a93a](https://github.com/meridianlabs-ai/inspect_sentinel/commit/9c8a93a1fbae177e8f7226494f31df8b3aaf1d0e))
* rename final() to decide_final() ([#18](https://github.com/meridianlabs-ai/inspect_sentinel/issues/18)) ([b9441ce](https://github.com/meridianlabs-ai/inspect_sentinel/commit/b9441ce82ed10a13bcbab0efe0fa9bbf6b8651db))
* report types (Observation, Decision, Reported) ([2c61f09](https://github.com/meridianlabs-ai/inspect_sentinel/commit/2c61f09b8ddc4aef34114fbf2e547638d6f107a2))
* run a protocol's monitors and child protocols concurrently ([#5](https://github.com/meridianlabs-ai/inspect_sentinel/issues/5)) ([686e27b](https://github.com/meridianlabs-ai/inspect_sentinel/commit/686e27b5d2e357d4ed0e6d1e09b506176958e8b2))
* separate model and role on Host.generate, and Context docs that match what fills ([#23](https://github.com/meridianlabs-ai/inspect_sentinel/issues/23)) ([fc387c7](https://github.com/meridianlabs-ai/inspect_sentinel/commit/fc387c775a81a455bb73d9c6ee8e58af421e0ca7))
* sequential(), an ordered composition that hands escalations forward ([#33](https://github.com/meridianlabs-ai/inspect_sentinel/issues/33)) ([4f2c182](https://github.com/meridianlabs-ai/inspect_sentinel/commit/4f2c1826dfaa6b6480d2f2bb067adabcc84d3b9d))
* threshold rejects without telling the agent why, and explains terminations in the log ([#15](https://github.com/meridianlabs-ai/inspect_sentinel/issues/15)) ([a5c3992](https://github.com/meridianlabs-ai/inspect_sentinel/commit/a5c3992e73062fda3e5d1d73470622fa115b1374))
* tool-stage step payloads ([cd885c0](https://github.com/meridianlabs-ai/inspect_sentinel/commit/cd885c0a17c7503b53a93fa83e473178ca15319b))


### Bug Fixes

* a modify may rewrite only the call's arguments ([#13](https://github.com/meridianlabs-ai/inspect_sentinel/issues/13)) ([2d181a9](https://github.com/meridianlabs-ai/inspect_sentinel/commit/2d181a970055c67a81a412b794ba6f1695bd5df5))
* assert step lists by identity ([7448a01](https://github.com/meridianlabs-ai/inspect_sentinel/commit/7448a01327a00aa1fc83599955e190721c86d00e))
* covariant Reported and firmer report tests ([6bdcbd9](https://github.com/meridianlabs-ai/inspect_sentinel/commit/6bdcbd9a404aec7970594e0292606f87cfa35c01))
* document every step field and test input/history independently ([12eb565](https://github.com/meridianlabs-ai/inspect_sentinel/commit/12eb5651234543a265eff3de025d088a8b608688))
* namespace the root store by path and record all recorder arguments ([6fc8fc5](https://github.com/meridianlabs-ai/inspect_sentinel/commit/6fc8fc5cd41d47629ab12e29a7f9fa9964c682e4))
