# WebUI — cross-platform manual test console

`src/QwenZImage.NET.WebUI` is an optional **ASP.NET Core Minimal API** app for
manually testing the library. It reproduces the functional surface of the
original Go `h3webui` (health, presets, serial job queue, cancel, retry,
progress, output gallery, trash/restore, unload, SSE) but is written entirely
in `.NET 10` — **no Go code, no Go process**. By default the browser talks to
this app and the app calls `QziClient` → `qzi` C ABI → native backend directly
in-process, with no HTTP hop.

Because the bundled native library is only a deterministic **reference backend
that emits noise**, the WebUI has one extra, **strictly opt-in** transport: set
`QZI_ENGINE_URL` (or `--engine`) and it will talk to a real `h3 serve`
C/CUDA engine over HTTP instead. That adapter lives only in this test console,
is pure `HttpClient` + `System.Text.Json`, and does **not** change the library
or the NuGet package, which remain pure Native P/Invoke.

It is a **test console, not the shipping SDK**. It is not packed into the
NuGet package (`IsPackable=false`); the NuGet deliverable is the library plus
its two native runtimes.

```
browser ──HTTP + SSE──▶ QwenZImage.NET.WebUI (ASP.NET Core, :7860)
                            │  QziGenerationQueue (Channels, single reader)
                            │
           ┌────────────────┴─────────────────────────────────┐
           │ default: QziClient (LibraryImport P/Invoke)       │  QZI_ENGINE_URL unset
           │   qzi.dll/libqzi.so → qzi_backend                 │  → noise-only reference
           │                                                  │
           │ optional: H3ServeExecutor (HttpClient)            │  QZI_ENGINE_URL set
           │   POST /generate · GET /progress · /health        │  → real images
           └──────────────────────────────────────────────────┘
```

## Run it

Prerequisites: .NET 10 SDK and a built `qzi` native library (see the root
README). On Windows the library is `qzi.dll`, on Linux `libqzi.so`.

```bash
# Windows (PowerShell)
.\scripts\build-native.ps1
dotnet run --project src\QwenZImage.NET.WebUI -c Release

# Linux / macOS
bash scripts/build-native-linux.sh
dotnet run --project src/QwenZImage.NET.WebUI -c Release
```

Then open <http://127.0.0.1:7860/>. For a deployable directory (contains
`wwwroot/` plus both staged native runtimes, so no `QZI_NATIVE_LIBRARY` is
needed):

```bash
dotnet publish src/QwenZImage.NET.WebUI -c Release -o artifacts/webui-publish
dotnet artifacts/webui-publish/QwenZImage.NET.WebUI.dll
```

The app falls back to its own directory for static assets when the launch
directory has no `wwwroot/`, so the published build can be started from any
working directory. User data (models, outputs) still resolves relative to the
launch directory.

Override the listen address with `QZI_WEBUI_URLS` or the standard `--urls` /
`ASPNETCORE_URLS` (command line always wins):

```bash
QZI_WEBUI_URLS=http://0.0.0.0:8080 dotnet run --project src/QwenZImage.NET.WebUI -c Release
```

> Binding to a non-loopback address exposes the console to your network. It is
> intended for local manual testing; there is no authentication.

## Real images: connect a running `h3 serve`

The bundled `qzi.dll` / `libqzi.so` is a deterministic test stub — its
`make_rgb()` writes pseudo-random pixels and its `dit` loop only advances a
progress bar. **Matching the `steps` of the original Go version cannot fix
that**, because no diffusion happens at all. The real C/CUDA engine and the
model weights live in the original `h3-spark.c` project; start its server and
point this WebUI at it:

```bash
# Terminal 1 — the real engine (use your own model directories)
h3 serve --port 8411 --qwen21-dir /path/to/qwen21 --zimage-dir /path/to/zimage

# Terminal 2 — this WebUI, now showing real images
QZI_ENGINE_URL=http://127.0.0.1:8411 \
  dotnet run --project src/QwenZImage.NET.WebUI -c Release

# Equivalent CLI switch (mirrors the original Go --engine flag)
dotnet run --project src/QwenZImage.NET.WebUI -c Release -- --engine http://127.0.0.1:8411
```

With the variable set, the bundled native library is **never loaded**.
`/api/health` reports `transport: "h3-serve-http"`,
`engine.backend: "h3-serve"`, `engine.reference_backend: false`, and the yellow
"reference backend — noise" banner disappears. Every other feature (presets,
serial queue, cancel, retry, SSE progress, gallery, trash, unload) is unchanged.

`H3ServeExecutor` maps the strongly typed request onto the C engine protocol
documented in `h3-spark.c/docs/PROTOCOL.md`:

| WebUI / library field | `h3 serve` behaviour |
| --- | --- |
| `QwenImageOptions` (turbo) | `model=qwen21`, `tier=turbo`, `steps=8`, literal `guidance=1.0`; **no** `negative_prompt` |
| `QwenImageOptions` (base) | plus `steps`, `guidance`, optional `negative_prompt`, `lora` + `lora_scale` |
| `ZImageOptions` | `model=zimage` + `checkpoint`; **never** `steps`/`guidance`/`frames`/`pdd8`/`tier` |
| PNG bytes | engine response body → `GenerationResult.FromPng(...)` → queue output sink → gallery, byte-for-byte |
| `GET /progress` | polled every 500 ms, mapped to `GenerationPhase` and re-published as SSE `progress` |
| `GET /health` / `POST /unload` | mapped to `/api/health` and `/api/unload` |
| HTTP 400 / 503 / 500 | mapped to `QziStatus.InvalidArgument` / `Busy` or `NotConfigured` / `EngineFailure`; surfaced on the job |

The engine is single-flight and its `/generate` is a synchronous long request
(minutes). The WebUI queue already serialises jobs, so the C engine's 503
`busy` is not normally seen. Set `QZI_ENGINE_TIMEOUT_MIN` high enough for cold
model loads; the default is 30 minutes. Only `/generate` uses that long
deadline — the control-plane calls (`/health`, `/unload`, `/progress`) each
have a 5-second timeout, so a wedged engine surfaces as `ok=false` in
`/api/health` instead of hanging the console.

For a **no-GPU protocol smoke test**, use the bundled fixture instead of a real
engine:

```bash
python3 scripts/mock-h3-serve.py --port 18411          # fixed 27-byte PNG
QZI_ENGINE_URL=http://127.0.0.1:18411 \
  dotnet run --project src/QwenZImage.NET.WebUI -c Release
```

CI uses exactly this pair to assert that the engine's bytes arrive unchanged in
the output gallery. See [ENGINE_INTEGRATION.md](ENGINE_INTEGRATION.md) for the
long-term option of implementing a real native `qzi_backend`, which would let
the library itself (not just the WebUI) produce real images.

### Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `QZI_WEBUI_URLS` | `http://127.0.0.1:7860` | Kestrel listen addresses (`;`-separated) |
| `QZI_WEBUI_OUTPUT` | `<cwd>/outputs` | Gallery / trash directory |
| `QZI_MODEL_DIR` | `<cwd>` | Passed to `QziClientOptions.Qwen21Directory` / `ZImageDirectory` |
| `QZI_MODEL_CACHE` | `<temp>/qzi-text-cache` | Text-encoder cache directory |
| `QZI_NATIVE_LIBRARY` | auto-probe | Full path to `qzi.dll` / `libqzi.so` (ignored in HTTP mode) |
| `QZI_DEVICE_ID` | `-1` | CUDA device selector passed to the native backend |
| `QZI_ENGINE_URL` | unset | Base URL of a real `h3 serve`; **enables the HTTP transport and real images** |
| `--engine URL` | — | CLI equivalent of `QZI_ENGINE_URL`, takes precedence over the environment |
| `QZI_ENGINE_TIMEOUT_MIN` | `30` | Whole-request deadline for `/generate` (cold model loads are slow) |
| `QZI_ENGINE_PROGRESS_MS` | `500` | Poll interval for the engine's optional `GET /progress` |

The WebUI starts even if the native library cannot be loaded: `/api/health`
then reports `ok=false` with the exact error, and `/api/events` emits an
`event: error`. This makes it usable to diagnose a missing runtime. In HTTP
mode, an unreachable engine similarly shows up as `ok=false` with the
connection error in `/api/health`.

## HTTP API

All responses are JSON with **snake_case** field names, matching the original
Go WebUI wire format. Errors use `{"error": "...", "detail": "..."}`.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | WebUI + native engine status, output/model dirs, uptime |
| `GET` | `/api/presets` | `{"presets":[...]}` — all `QziPresets` with a `model` discriminator |
| `POST` | `/api/jobs` | Validate + enqueue; `202` with the job snapshot |
| `GET` | `/api/jobs` | `{"jobs":[...]}` newest first; optional `?client=` filter |
| `GET` | `/api/jobs/{id}` | One job, `404` if unknown |
| `POST` | `/api/jobs/{id}/cancel` | Cancel queued job (engine never called) or signal a running one; `409` if terminal |
| `POST` | `/api/jobs/{id}/retry` | Re-submit the *same* request object; `202`, `409` if not terminal |
| `GET` / `POST` | `/api/unload` | Release the resident model (native `qzi_engine_unload` or engine `POST /unload`) |
| `GET` | `/api/outputs` | Gallery listing, newest first |
| `GET` | `/api/outputs/{name}` | Serve one file with range support |
| `DELETE` | `/api/outputs/{name}` | Move one file into `.trash` |
| `DELETE` | `/api/outputs` | Batch delete; body `{"names":[...]}` |
| `POST` | `/api/outputs/restore` | Batch restore from `.trash`; never overwrites a newer file |
| `GET` | `/api/events` | Server-sent events: `job`, `progress`, `error`, 15 s heartbeat |
| `GET` | `/` | Static single-page console |

### Request envelope (`POST /api/jobs`)

```json
{
  "model": "qwen21",
  "prompt": "a quiet mountain lake at dawn",
  "width": 1024,
  "height": 1024,
  "seed": 42,
  "tier": "turbo",
  "client": "manual-test"
}
```

`model` accepts `qwen21` (aliases `qwen`, `qwen-image`) or `zimage`
(aliases `z-image`, `z_image`). Qwen21 adds `tier` (`turbo`/`base`), and for
`base`: `steps`, `guidance`, `negative_prompt`, `lora`, `lora_scale`.
Z-Image adds `checkpoint` (`bf16`, `nunchaku`, `nunchaku-r<rank>`).
The full validation rules are in [PARAMETERS.md](PARAMETERS.md); invalid
envelopes fail with `400` **before** any native model loading.

### SSE (`/api/events`)

```
retry: 3000

event: job
data: {"id":1,"state":"running","model":"zimage", ...}

event: progress
data: {"job_id":1,"phase":"Dit","completed":3,"total":8,"fraction":0.375, ...}

: ping
```

`job` events are emitted on every state transition and progress tick. While a
job is running, a matching `progress` event is emitted as well. The stream
sends a comment heartbeat every 15 seconds so proxies do not close idle
connections. `EventSource` reconnects automatically.

## Front-end features

- Model selector, presets filtered by model, prompt / negative prompt, width,
  height (step hint changes 32 ↔ 16), seed with randomiser, tier / steps /
  guidance / LoRA / checkpoint fields with automatic enable-disable rules.
- Live job table with state badge, progress bar, elapsed time, cancel / retry.
- Output gallery with lazy thumbnails, multi-select, batch delete, "Undo
  delete" (restore), lightbox, and direct link to the original PNG.
- Health pill showing resident model / backend (`qzi-reference` vs `h3-serve`), plus an "Unload model" action.
- A yellow banner whenever the noise-only reference backend is active, with the exact `QZI_ENGINE_URL` command to switch to real inference.

## Output safety

Mirrors the Go WebUI rules in `OutputStore`:

- Names must be bare file names: no `/`, `\`, NUL, `.`, `..`, or leading `.`.
- Delete **moves** the file to `<out>/.trash` instead of unlinking it.
- Restore refuses symlinks and, if the original name has been reused, writes
  `<stem>-restored-<i><ext>` rather than overwriting the newer file.
- Only the top-level directory is enumerated; `.trash` is a subdirectory and
  therefore never appears in the gallery.

## Cross-platform notes

- The project targets `net10.0` with `Microsoft.NET.Sdk.Web` and has no
  platform-specific code or native dependency of its own; the only native
  library is `qzi`, resolved by `NativeLibraryLoader`.
- The csproj copies `artifacts/runtimes/{win-x64,linux-x64}/native/*` into
  both build and publish output. That makes the console self-contained on
  Windows and Linux; `QZI_NATIVE_LIBRARY` remains available to override it
  with a real CUDA engine.
- Static assets live in `wwwroot/` and are served by `UseStaticFiles` on both
  Windows and Linux. No Node/npm build step is required.
- CI builds the solution on `ubuntu-latest` and `windows-latest` and runs two
  runtime smoke tests on Linux: one against the in-process reference backend,
  one against `scripts/mock-h3-serve.py` over the HTTP transport (verifying
  `reference_backend=false` and byte-identical engine output in the gallery).
- On Windows, if the default port is in a system-reserved range, set
  `QZI_WEBUI_URLS` to a free port (for example
  `http://127.0.0.1:51234`).
