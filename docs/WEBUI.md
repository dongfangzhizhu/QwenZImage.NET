# WebUI — cross-platform manual test console

`src/QwenZImage.NET.WebUI` is an optional **ASP.NET Core Minimal API** app for
manually testing the library. It reproduces the functional surface of the
original Go `h3webui` (health, presets, serial job queue, cancel, retry,
progress, output gallery, trash/restore, unload, SSE) but is written entirely
in `.NET 10` — there is **no Go process, no sidecar, and no HTTP hop to a
separate engine server**. The browser talks to this app; this app calls
`QziClient` → `qzi` C ABI → native backend directly in-process.

It is a **test console, not the shipping SDK**. It is not packed into the
NuGet package (`IsPackable=false`); the NuGet deliverable is the library plus
its two native runtimes.

```
browser ──HTTP + SSE──▶ QwenZImage.NET.WebUI (ASP.NET Core, :7860)
                            │  QziGenerationQueue (Channels, single reader)
                            ▼  QziClient (LibraryImport P/Invoke)
                         qzi.dll / libqzi.so  ──▶ qzi_backend (reference or real engine)
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

### Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `QZI_WEBUI_URLS` | `http://127.0.0.1:7860` | Kestrel listen addresses (`;`-separated) |
| `QZI_WEBUI_OUTPUT` | `<cwd>/outputs` | Gallery / trash directory |
| `QZI_MODEL_DIR` | `<cwd>` | Passed to `QziClientOptions.Qwen21Directory` / `ZImageDirectory` |
| `QZI_MODEL_CACHE` | `<temp>/qzi-text-cache` | Text-encoder cache directory |
| `QZI_NATIVE_LIBRARY` | auto-probe | Full path to `qzi.dll` / `libqzi.so` |
| `QZI_DEVICE_ID` | `-1` | CUDA device selector passed to the backend |

The WebUI starts even if the native library cannot be loaded: `/api/health`
then reports `ok=false` with the exact error, and `/api/events` emits an
`event: error`. This makes it usable to diagnose a missing runtime.

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
| `GET` / `POST` | `/api/unload` | Release the resident model via `QziClient.Unload()` |
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
- Health pill showing resident model / backend, plus an "Unload model" action.

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
- CI builds the solution on `ubuntu-latest` and `windows-latest` and runs a
  runtime smoke test of health/presets/jobs/outputs on Linux.
- On Windows, if the default port is in a system-reserved range, set
  `QZI_WEBUI_URLS` to a free port (for example
  `http://127.0.0.1:51234`).
