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
  // Qwen-Image 2.1 edit state. Both are PNG data URLs produced by a canvas;
  // the server decodes them into planar rasters (no imaging dependency).
  referencePng: null,
  maskDirty: false,
  editingAvailable: true,
  // Z-Image Turbo Fun ControlNet Union state. The control image is a PNG data
  // URL re-encoded by a canvas; it must already be preprocessed upstream.
  controlPng: null,
  controlnetAvailable: true,
  drawing: false,
  lastPoint: null,
};

const modelFamily = () => ($("model").value.startsWith("qwen21") ? "qwen21" : "zimage");
const isEditMode = () => $("model").value === "qwen21-edit";
const isControlMode = () => $("model").value === "zimage-control";

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
        `config: ${h.config_path || "(defaults)"}`,
        `out_dir: ${h.out_dir}`,
        `model_dir: ${h.model_dir}`,
        `native: ${h.native_library || "(runtime default probe)"}`,
        `native_source: ${h.native_library_source || "unknown"}`,
        `served: ${h.engine?.served ?? 0}`,
      ].join("\n");
      $("backend-banner").hidden = !reference;
      // Editing is additive in ABI v1: an older native library simply has no
      // qzi_generate_qwen21_edit export. Do not offer a mode that would fail.
      state.editingAvailable = h.supports_image_editing === true;
      const editOption = document.querySelector('#model option[value="qwen21-edit"]');
      if (editOption) editOption.disabled = !state.editingAvailable;
      if (!state.editingAvailable && isEditMode()) {
        $("model").value = "qwen21";
        switchModel();
      }
      // ControlNet is additive in ABI v1 too: hide the mode when the loaded
      // library has no qzi_generate_zimage_control export, so a job never
      // silently degrades to text-to-image or the noise backend.
      state.controlnetAvailable = h.supports_controlnet === true;
      const controlOption = document.querySelector('#model option[value="zimage-control"]');
      if (controlOption) controlOption.disabled = !state.controlnetAvailable;
      if (!state.controlnetAvailable && isControlMode()) {
        $("model").value = "zimage";
        switchModel();
      }
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
  const family = modelFamily();
  const select = $("preset");
  select.innerHTML = '<option value="">Custom</option>';
  state.presets.forEach((preset, index) => {
    if (preset.model !== family) return;
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
  if (modelFamily() === "qwen21") {
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
  const family = modelFamily();
  const edit = isEditMode();
  const step = family === "qwen21" ? 32 : 16;
  $("width").step = String(step);
  $("height").step = String(step);
  $("width").min = String(step);
  $("height").min = String(step);
  $("width-hint").textContent = `multiple of ${step}`;
  $("height-hint").textContent = `multiple of ${step}`;

  // "qwen21" fields are shared by t2i and edit; "qwen21-edit" adds the
  // reference/mask controls on top of them. "zimage-control" is a zimage
  // variant (so checkpoint / 16-px steps still apply) that adds the control
  // image, type and scale controls.
  document.querySelectorAll("[data-only]").forEach((el) => {
    const only = el.getAttribute("data-only");
    el.hidden = only === "qwen21" ? family !== "qwen21"
      : only === "qwen21-edit" ? !edit
      : only === "zimage" ? family !== "zimage"
      : only === "zimage-control" ? !isControlMode()
      : false;
  });

  const isBase = family === "qwen21" && $("tier").value === "base";
  document.querySelectorAll("[data-requires='base']").forEach((el) => {
    el.querySelectorAll("input, select, textarea").forEach((input) => { input.disabled = !isBase; });
  });
  $("negative").disabled = !isBase;
  syncMaskControls();
  syncControlScaleHint();
}

function switchModel() {
  const step = modelFamily() === "qwen21" ? 32 : 16;
  ["width", "height"].forEach((id) => {
    const input = $(id);
    const value = Math.min(2048, Math.max(step, Number(input.value) || 1024));
    input.value = String(Math.round(value / step) * step);
  });
  renderPresets();
  $("preset").value = "";
  syncModelUI();
}

// ------------------------------------------------------------- image edits

/* The native decoder accepts 8-bit PNG, so re-encode every upload through a
 * canvas. That also gives us a single resolution for both the reference and
 * the painted mask, which the engine cover-fits with the SAME crop. */
function syncMaskControls() {
  const masking = $("masking-strength");
  if (!masking) return;
  masking.disabled = !state.maskDirty;
  $("masking-strength-hint").textContent = Number(masking.value).toFixed(2);
  $("strength-hint").textContent = Number($("strength").value).toFixed(2);
  $("brush-size-hint").textContent = `${$("brush-size").value} px`;
}

function clearMask() {
  const canvas = $("mask-canvas");
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  state.maskDirty = false;
  syncMaskControls();
}

function fillMask() {
  const canvas = $("mask-canvas");
  const ctx = canvas.getContext("2d");
  ctx.save();
  ctx.globalCompositeOperation = "source-over";
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.restore();
  state.maskDirty = true;
  syncMaskControls();
}

function removeReference() {
  state.referencePng = null;
  state.maskDirty = false;
  $("reference-file").value = "";
  $("reference-preview").removeAttribute("src");
  $("mask-editor").hidden = true;
  $("reference-empty").hidden = false;
  syncMaskControls();
}

async function loadReference(file) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();

    // Mirror qzi.c / QziRasterImage limits: <= 4096 per side and at most
    // 2048*2048 total pixels. Downscale before encoding so the server never
    // has to reject a perfectly good photo.
    const maxSide = 4096, maxPixels = 2048 * 2048;
    const nativeW = image.naturalWidth, nativeH = image.naturalHeight;
    const scale = Math.min(1, maxSide / nativeW, maxSide / nativeH,
                           Math.sqrt(maxPixels / (nativeW * nativeH)));
    const width = Math.max(1, Math.round(nativeW * scale));
    const height = Math.max(1, Math.round(nativeH * scale));

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").drawImage(image, 0, 0, width, height);
    state.referencePng = canvas.toDataURL("image/png");

    $("reference-preview").src = state.referencePng;
    const mask = $("mask-canvas");
    mask.width = width;
    mask.height = height;
    clearMask();
    $("reference-empty").hidden = true;
    $("mask-editor").hidden = false;
    showToast(`Reference loaded (${width}×${height}). Paint white to regenerate a region.`);
  } catch (error) {
    showToast(`Could not read the reference image: ${error.message || error}`);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function strokeGeometry(event) {
  const canvas = $("mask-canvas");
  const rect = canvas.getBoundingClientRect();
  const ratio = canvas.width / rect.width;
  return {
    ctx: canvas.getContext("2d"),
    ratio,
    x: (event.clientX - rect.left) * ratio,
    y: (event.clientY - rect.top) * ratio,
    width: Number($("brush-size").value) * ratio,
  };
}

function prepareStroke(ctx, width) {
  const erasing = $("brush-mode").value === "erase";
  ctx.save();
  ctx.globalCompositeOperation = erasing ? "destination-out" : "source-over";
  ctx.strokeStyle = "#ffffff";
  ctx.fillStyle = "#ffffff";
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}

function beginStroke(event) {
  if ($("mask-editor").hidden) return;
  event.preventDefault();
  const g = strokeGeometry(event);
  state.drawing = true;
  state.lastPoint = { x: g.x, y: g.y };
  $("mask-canvas").setPointerCapture(event.pointerId);
  prepareStroke(g.ctx, g.width);
  // A single click still paints a dot, not an invisible zero-length line.
  g.ctx.beginPath();
  g.ctx.arc(g.x, g.y, g.width / 2, 0, Math.PI * 2);
  g.ctx.fill();
  g.ctx.restore();
  state.maskDirty = true;
  syncMaskControls();
}

function moveStroke(event) {
  if (!state.drawing) return;
  event.preventDefault();
  const g = strokeGeometry(event);
  prepareStroke(g.ctx, g.width);
  g.ctx.beginPath();
  g.ctx.moveTo(state.lastPoint.x, state.lastPoint.y);
  g.ctx.lineTo(g.x, g.y);
  g.ctx.stroke();
  g.ctx.restore();
  state.lastPoint = { x: g.x, y: g.y };
}

function endStroke() {
  state.drawing = false;
  state.lastPoint = null;
}

// Transparent pixels decode to mask value 0 ("keep"); opaque white strokes
// decode to 255 ("regenerate"). Only send the mask once it has been painted.
function maskPng() {
  return state.maskDirty ? $("mask-canvas").toDataURL("image/png") : null;
}

// ------------------------------------------ Z-Image ControlNet Union

/* The control image must already be preprocessed. Re-encode every upload
 * through a canvas to 8-bit PNG (the only format the native decoder accepts)
 * and downscale to the QziRasterImage limits. No server-side estimator. */
async function loadControlImage(file) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    const maxSide = 4096, maxPixels = 2048 * 2048;
    const nativeW = image.naturalWidth, nativeH = image.naturalHeight;
    const scale = Math.min(1, maxSide / nativeW, maxSide / nativeH,
                           Math.sqrt(maxPixels / (nativeW * nativeH)));
    const width = Math.max(1, Math.round(nativeW * scale));
    const height = Math.max(1, Math.round(nativeH * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").drawImage(image, 0, 0, width, height);
    state.controlPng = canvas.toDataURL("image/png");
    $("control-preview").src = state.controlPng;
    $("control-empty").hidden = true;
    $("control-editor").hidden = false;
    showToast(`Control image loaded (${width}×${height}). Used as-is (no preprocessing).`);
  } catch (error) {
    showToast(`Could not read the control image: ${error.message || error}`);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function removeControlImage() {
  state.controlPng = null;
  $("control-file").value = "";
  $("control-preview").removeAttribute("src");
  $("control-editor").hidden = true;
  $("control-empty").hidden = false;
}

function syncControlScaleHint() {
  const slider = $("control-scale");
  if (!slider) return;
  const value = Number(slider.value);
  $("control-scale-hint").textContent = value === 0 ? "0 (engine default 0.65)" : value.toFixed(2);
}

// --------------------------------------------------------------------- form

function buildRequestBody() {
  const model = $("model").value;
  const family = modelFamily();
  const body = {
    model,
    prompt: $("prompt").value.trim(),
    width: Number($("width").value),
    height: Number($("height").value),
    seed: Number($("seed").value),
    client: "webui",
  };
  if (family === "qwen21") {
    body.tier = $("tier").value;
    // LoRA is valid on both tiers in the library / C ABI; only the sampler
    // controls and negative prompt are base-only.
    const lora = $("lora").value.trim();
    if (lora) {
      body.lora = lora;
      body.lora_scale = Number($("lora-scale").value);
    }
    if (body.tier === "base") {
      body.steps = Number($("steps").value);
      body.guidance = Number($("guidance").value);
      const negative = $("negative").value.trim();
      if (negative) body.negative_prompt = negative;
    }
  } else {
    body.checkpoint = $("checkpoint").value;
  }
  if (isEditMode()) {
    body.reference_image = state.referencePng;
    body.strength = Number($("strength").value);
    const mask = maskPng();
    if (mask) {
      body.mask_image = mask;
      // masking_strength would be silently dropped without a mask, so the
      // mapper rejects that combination. Only send both together.
      body.masking_strength = Number($("masking-strength").value);
    }
  }
  if (isControlMode()) {
    body.control_image = state.controlPng;
    body.control_type = $("control-type").value;
    body.control_scale = Number($("control-scale").value);
  }
  return body;
}

async function submit() {
  if (!$("prompt").value.trim()) {
    showToast("Please enter a prompt.");
    $("prompt").focus();
    return;
  }
  if (isEditMode() && !state.referencePng) {
    showToast("Upload a reference image before queueing an edit.");
    $("reference-file").focus();
    return;
  }
  if (isControlMode() && !state.controlPng) {
    showToast("Upload a preprocessed control image before queueing a ControlNet job.");
    $("control-file").focus();
    return;
  }
  const payload = JSON.stringify(buildRequestBody());
  // The API caps the body at 64 MiB. Base64 inflates by ~4/3, so a very large
  // photo plus mask / control image can exceed it; say so before it is wasted.
  if (payload.length > 60 * 1024 * 1024) {
    showToast("Reference / mask / control image payload is too large (60 MiB limit). Use a smaller image.");
    return;
  }
  const button = $("generate");
  button.disabled = true;
  try {
    const job = await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
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
    const editBadge = job.edit
      ? `<span class="job-edit-badge">edit${job.has_mask ? " · mask" : ""}</span>`
      : "";
    const controlBadge = job.control
      ? `<span class="job-control-badge">control</span>`
      : "";
    const actions = [];
    if (!terminal) actions.push(`<button class="btn btn-ghost btn-sm" data-action="cancel" data-id="${job.id}">Cancel</button>`);
    else actions.push(`<button class="btn btn-ghost btn-sm" data-action="retry" data-id="${job.id}">Retry</button>`);
    return `<tr>
      <td>#${job.id}${job.retry_of ? `<div class="hint">↻ #${job.retry_of}</div>` : ""}</td>
      <td>${escapeHtml(job.model)}${editBadge}${controlBadge}<div class="job-prompt" title="${escapeHtml(job.prompt)}">${escapeHtml(job.prompt)}</div></td>
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

  // Image editing: reference upload, brush mask and the two strength sliders.
  $("reference-file").addEventListener("change", (event) => loadReference(event.target.files?.[0]));
  $("mask-clear").addEventListener("click", clearMask);
  $("mask-fill").addEventListener("click", fillMask);
  $("reference-remove").addEventListener("click", removeReference);
  const maskCanvas = $("mask-canvas");
  maskCanvas.addEventListener("pointerdown", beginStroke);
  maskCanvas.addEventListener("pointermove", moveStroke);
  maskCanvas.addEventListener("pointerup", endStroke);
  maskCanvas.addEventListener("pointercancel", endStroke);
  maskCanvas.addEventListener("lostpointercapture", endStroke);
  ["strength", "masking-strength", "brush-size"].forEach((id) => {
    $(id).addEventListener("input", syncMaskControls);
  });

  // Z-Image ControlNet Union: control-image upload, removal and scale slider.
  $("control-file").addEventListener("change", (event) => loadControlImage(event.target.files?.[0]));
  $("control-remove").addEventListener("click", removeControlImage);
  $("control-scale").addEventListener("input", syncControlScaleHint);
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
