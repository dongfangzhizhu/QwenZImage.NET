/* QwenZImage.NET manual test console.
 * Wire format is snake_case JSON, matching the original Go WebUI API.
 * Everything here is plain browser JS: no build step, no Go, no dependencies.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  presets: [],
  jobs: new Map(),
  outputs: [],
  selection: new Set(),
  lastDeleted: [],
  lightboxUrl: null,
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { body = { error: text }; }
  if (!response.ok) {
    throw new Error(body?.detail ? `${body.error}: ${body.detail}` : (body?.error || `HTTP ${response.status}`));
  }
  return body;
}

function showToast(message) {
  $("toast-text").textContent = message;
  $("toast").hidden = false;
}
function hideToast() { $("toast").hidden = true; }

// ------------------------------------------------------------------ health

async function refreshHealth() {
  try {
    const h = await api("/api/health");
    const pill = $("health-pill");
    if (h.ok) {
      const reference = h.engine?.reference_backend === true;
      const backend = reference ? "reference backend (noise)" : (h.engine?.backend || "engine backend");
      pill.className = reference ? "pill pill-wait" : "pill pill-ok";
      pill.textContent = `● ${backend} · ${h.engine?.resident_model || "no model"}`;
      pill.title = [
        `transport: ${h.transport}`,
        `engine_url: ${h.engine_url || "(in-process native)"}`,
        `out_dir: ${h.out_dir}`,
        `model_dir: ${h.model_dir}`,
        `served: ${h.engine?.served ?? 0}`,
      ].join("\n");
      $("backend-banner").hidden = !reference;
    } else {
      $("backend-banner").hidden = true;
      pill.className = "pill pill-bad";
      pill.textContent = "● engine unavailable";
      pill.title = h.error || "unknown error";
    }
  } catch (error) {
    $("health-pill").className = "pill pill-bad";
    $("health-pill").textContent = "● server unreachable";
    showToast(String(error.message || error));
  }
}

// ----------------------------------------------------------------- presets

async function refreshPresets() {
  try {
    state.presets = (await api("/api/presets")).presets || [];
    renderPresets();
  } catch (error) { showToast(String(error.message || error)); }
}

function renderPresets() {
  const model = $("model").value;
  const select = $("preset");
  select.innerHTML = '<option value="">Custom</option>';
  state.presets.forEach((preset, index) => {
    if (preset.model !== model) return;
    const option = document.createElement("option");
    option.value = String(state.presets.indexOf(preset));
    option.textContent = `${preset.label}${preset.model === "qwen21" ? ` · ${preset.tier || "turbo"}` : ""}`;
    option.dataset.index = String(index);
    select.appendChild(option);
  });
}

function applyPreset() {
  const raw = $("preset").value;
  if (raw === "") return;
  const preset = state.presets[Number(raw)];
  if (!preset) return;
  $("width").value = preset.width;
  $("height").value = preset.height;
  if (preset.model === "qwen21") {
    $("tier").value = preset.tier || "turbo";
    $("steps").value = preset.steps ?? 8;
    $("guidance").value = preset.guidance ?? 1;
    $("lora").value = preset.lora || "";
    $("lora-scale").value = preset.lora_scale ?? 1;
  } else {
    $("checkpoint").value = preset.checkpoint || "bf16";
  }
  syncModelUI();
}

function markCustom() { $("preset").value = ""; }

// -------------------------------------------------------------- model rules

function syncModelUI() {
  const model = $("model").value;
  const step = model === "qwen21" ? 32 : 16;
  $("width").step = String(step);
  $("height").step = String(step);
  $("width").min = String(step);
  $("height").min = String(step);
  $("width-hint").textContent = `multiple of ${step}`;
  $("height-hint").textContent = `multiple of ${step}`;

  document.querySelectorAll("[data-only]").forEach((el) => {
    el.hidden = el.getAttribute("data-only") !== model;
  });

  const isBase = model === "qwen21" && $("tier").value === "base";
  document.querySelectorAll("[data-requires='base']").forEach((el) => {
    el.querySelectorAll("input, select, textarea").forEach((input) => { input.disabled = !isBase; });
  });
  $("negative").disabled = !isBase;
}

function switchModel() {
  const step = $("model").value === "qwen21" ? 32 : 16;
  ["width", "height"].forEach((id) => {
    const input = $(id);
    const value = Math.min(2048, Math.max(step, Number(input.value) || 1024));
    input.value = String(Math.round(value / step) * step);
  });
  renderPresets();
  $("preset").value = "";
  syncModelUI();
}

// --------------------------------------------------------------------- form

function buildRequestBody() {
  const model = $("model").value;
  const body = {
    model,
    prompt: $("prompt").value.trim(),
    width: Number($("width").value),
    height: Number($("height").value),
    seed: Number($("seed").value),
    client: "webui",
  };
  if (model === "qwen21") {
    body.tier = $("tier").value;
    if (body.tier === "base") {
      body.steps = Number($("steps").value);
      body.guidance = Number($("guidance").value);
      const negative = $("negative").value.trim();
      if (negative) body.negative_prompt = negative;
      const lora = $("lora").value.trim();
      if (lora) {
        body.lora = lora;
        body.lora_scale = Number($("lora-scale").value);
      }
    }
  } else {
    body.checkpoint = $("checkpoint").value;
  }
  return body;
}

async function submit() {
  if (!$("prompt").value.trim()) {
    showToast("Please enter a prompt.");
    $("prompt").focus();
    return;
  }
  const button = $("generate");
  button.disabled = true;
  try {
    const job = await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildRequestBody()),
    });
    state.jobs.set(job.id, job);
    renderJobs();
    await refreshOutputs();
  } catch (error) {
    showToast(String(error.message || error));
  } finally {
    button.disabled = false;
  }
}

// --------------------------------------------------------------------- jobs

async function refreshJobs() {
  try {
    const body = await api("/api/jobs");
    state.jobs.clear();
    (body.jobs || []).forEach((job) => state.jobs.set(job.id, job));
    renderJobs();
  } catch (error) { showToast(String(error.message || error)); }
}

function renderJobs() {
  const tbody = $("job-rows");
  const jobs = [...state.jobs.values()].sort((a, b) => b.id - a.id);
  $("jobs-empty").hidden = jobs.length > 0;
  tbody.innerHTML = jobs.map((job) => {
    const terminal = ["done", "error", "cancelled"].includes(job.state);
    const fraction = Math.round((job.progress_fraction || 0) * 100);
    const phase = job.phase ? `${job.phase} ${job.progress_completed}/${job.progress_total}` : "—";
    const output = job.output_url
      ? `<a class="link" href="${escapeHtml(job.output_url)}" target="_blank" rel="noopener">PNG (${job.output_length ?? 0} B)</a>`
      : (job.error ? `<span class="error-text" title="${escapeHtml(job.error)}">${escapeHtml(job.error)}</span>` : "—");
    const actions = [];
    if (!terminal) actions.push(`<button class="btn btn-ghost btn-sm" data-action="cancel" data-id="${job.id}">Cancel</button>`);
    else actions.push(`<button class="btn btn-ghost btn-sm" data-action="retry" data-id="${job.id}">Retry</button>`);
    return `<tr>
      <td>#${job.id}${job.retry_of ? `<div class="hint">↻ #${job.retry_of}</div>` : ""}</td>
      <td>${escapeHtml(job.model)}<div class="job-prompt" title="${escapeHtml(job.prompt)}">${escapeHtml(job.prompt)}</div></td>
      <td><span class="state state-${escapeHtml(job.state)}">${escapeHtml(job.state)}</span></td>
      <td class="progress-cell">
        <div class="progress-track"><div class="progress-fill" style="width:${job.state === "done" ? 100 : fraction}%"></div></div>
        <div class="progress-text">${terminal ? `${Math.round(job.elapsed_ms || 0)} ms` : escapeHtml(phase)}</div>
      </td>
      <td>${output}</td>
      <td><div class="row-actions">${actions.join("")}</div></td>
    </tr>`;
  }).join("");
}

async function jobAction(event) {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const { action, id } = button.dataset;
  try {
    const job = await api(`/api/jobs/${id}/${action}`, { method: "POST" });
    state.jobs.set(job.id, job);
    renderJobs();
  } catch (error) { showToast(String(error.message || error)); }
}

function connectEvents() {
  if (typeof EventSource === "undefined") {
    $("sse-pill").textContent = "SSE unsupported · polling";
    $("sse-pill").className = "pill pill-wait";
    return;
  }
  const source = new EventSource("/api/events");
  source.onopen = () => {
    $("sse-pill").textContent = "● live";
    $("sse-pill").className = "pill pill-ok";
  };
  source.onerror = (event) => {
    // A named "error" event from the server carries a JSON payload; a plain
    // connection error does not.
    if (event?.data) {
      try { showToast(JSON.parse(event.data).error || "engine error"); } catch { /* ignore */ }
      return;
    }
    $("sse-pill").textContent = "SSE reconnecting…";
    $("sse-pill").className = "pill pill-wait";
  };
  source.addEventListener("error", (event) => {
    if (!event.data) return;
    try {
      const body = JSON.parse(event.data);
      showToast(body.detail ? `${body.error}: ${body.detail}` : body.error);
    } catch { /* ignore */ }
  });
  source.addEventListener("job", (event) => {
    const job = JSON.parse(event.data);
    const wasTerminal = state.jobs.get(job.id)?.state;
    state.jobs.set(job.id, job);
    renderJobs();
    if (["done", "error", "cancelled"].includes(job.state) && wasTerminal !== job.state) refreshOutputs();
  });
  source.addEventListener("progress", (event) => {
    const progress = JSON.parse(event.data);
    const job = state.jobs.get(progress.job_id);
    if (!job) return;
    job.phase = progress.phase;
    job.progress_completed = progress.completed;
    job.progress_total = progress.total;
    job.progress_fraction = progress.fraction;
    renderJobs();
  });
}

// ------------------------------------------------------------------ outputs

async function refreshOutputs() {
  try {
    state.outputs = (await api("/api/outputs")).outputs || [];
    const valid = new Set(state.outputs.map((o) => o.name));
    [...state.selection].forEach((name) => { if (!valid.has(name)) state.selection.delete(name); });
    renderGallery();
  } catch (error) { showToast(String(error.message || error)); }
}

function renderGallery() {
  const gallery = $("gallery");
  $("gallery-empty").hidden = state.outputs.length > 0;
  gallery.innerHTML = state.outputs.map((output) => `
    <div class="tile ${state.selection.has(output.name) ? "selected" : ""}" data-name="${escapeHtml(output.name)}">
      <input type="checkbox" data-role="select" ${state.selection.has(output.name) ? "checked" : ""} aria-label="Select ${escapeHtml(output.name)}">
      <img src="${escapeHtml(output.url)}" alt="${escapeHtml(output.name)}" loading="lazy" data-role="preview">
      <div class="tile-meta">${escapeHtml(output.name)} · ${Math.round(output.size / 1024)} KB</div>
    </div>`).join("");
}

function onGalleryClick(event) {
  const tile = event.target.closest(".tile");
  if (!tile) return;
  const name = tile.dataset.name;
  const role = event.target.dataset.role;
  if (role === "select" || event.target.tagName === "INPUT") {
    if (event.target.checked) state.selection.add(name);
    else state.selection.delete(name);
    tile.classList.toggle("selected", event.target.checked);
    return;
  }
  const output = state.outputs.find((o) => o.name === name);
  if (!output) return;
  $("lightbox-image").src = output.url;
  $("lightbox-name").textContent = `${output.name} · ${Math.round(output.size / 1024)} KB`;
  state.lightboxUrl = output.url;
  $("lightbox").hidden = false;
}

function closeLightbox() {
  $("lightbox").hidden = true;
  $("lightbox-image").src = "";
  state.lightboxUrl = null;
}

async function deleteOutputs() {
  const names = [...state.selection];
  if (names.length === 0) { showToast("Select at least one output first."); return; }
  try {
    const body = await api("/api/outputs", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names }),
    });
    state.lastDeleted = body.deleted || [];
    state.selection.clear();
    if ((body.failed || []).length) showToast(`${body.failed.length} file(s) could not be deleted.`);
    await refreshOutputs();
  } catch (error) { showToast(String(error.message || error)); }
}

async function restoreOutputs() {
  if (state.lastDeleted.length === 0) { showToast("Nothing to restore in this session."); return; }
  try {
    const body = await api("/api/outputs/restore", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names: state.lastDeleted }),
    });
    state.lastDeleted = [];
    if ((body.failed || []).length) showToast(`${body.failed.length} file(s) could not be restored.`);
    await refreshOutputs();
  } catch (error) { showToast(String(error.message || error)); }
}

async function unloadEngine() {
  try {
    await api("/api/unload", { method: "POST" });
    await refreshHealth();
  } catch (error) { showToast(String(error.message || error)); }
}

// ------------------------------------------------------------------- wiring

function wire() {
  $("model").addEventListener("change", switchModel);
  $("preset").addEventListener("change", applyPreset);
  ["width", "height", "seed", "steps", "guidance", "lora", "lora-scale", "checkpoint"].forEach((id) => {
    $(id).addEventListener("input", markCustom);
    $(id).addEventListener("change", markCustom);
  });
  $("tier").addEventListener("change", () => { syncModelUI(); markCustom(); });
  $("seed-dice").addEventListener("click", () => {
    $("seed").value = String(crypto.getRandomValues(new Uint32Array(1))[0]);
  });
  $("generate").addEventListener("click", submit);
  $("jobs-refresh").addEventListener("click", refreshJobs);
  $("job-rows").addEventListener("click", jobAction);
  $("gallery-refresh").addEventListener("click", refreshOutputs);
  $("gallery").addEventListener("click", onGalleryClick);
  $("select-all").addEventListener("click", () => {
    state.outputs.forEach((o) => state.selection.add(o.name));
    renderGallery();
  });
  $("clear-selection").addEventListener("click", () => { state.selection.clear(); renderGallery(); });
  $("delete-selected").addEventListener("click", deleteOutputs);
  $("restore-selected").addEventListener("click", restoreOutputs);
  $("unload-btn").addEventListener("click", unloadEngine);
  $("lightbox-close").addEventListener("click", closeLightbox);
  $("lightbox-download").addEventListener("click", () => {
    if (state.lightboxUrl) window.open(state.lightboxUrl, "_blank", "noopener");
  });
  $("lightbox").addEventListener("click", (event) => {
    if (event.target.id === "lightbox") closeLightbox();
  });
  $("toast-close").addEventListener("click", hideToast);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeLightbox();
  });
}

async function boot() {
  wire();
  syncModelUI();
  await Promise.all([refreshHealth(), refreshPresets(), refreshJobs(), refreshOutputs()]);
  connectEvents();
  // Safety net for browsers/proxies that buffer SSE: reflect job state anyway.
  setInterval(() => {
    const busy = [...state.jobs.values()].some((j) => j.state === "queued" || j.state === "running");
    if (busy) refreshJobs();
  }, 3000);
  setInterval(refreshHealth, 30000);
}

boot();
