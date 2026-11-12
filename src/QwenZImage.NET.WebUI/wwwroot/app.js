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
  // Third additive ABI v1 increment: outpainting / LanPaint / RGBA.
  advancedEditingAvailable: true,
  // Z-Image Turbo Fun ControlNet Union state. `controlSourcePng` is the raw
  // browser upload; `controlPng` is the guide produced by /api/preprocess and
  // is the only raster ever posted to /api/jobs. The native C ABI keeps its
  // raw-raster contract — estimation is an application-layer WebUI concern.
  controlSourcePng: null,
  controlSourceWidth: 0,
  controlSourceHeight: 0,
  controlPng: null,
  controlWidth: 0,
  controlHeight: 0,
  controlMeta: null,
  controlnetAvailable: true,
  preprocessCapabilities: null,
  preprocessTouched: false,
  preprocessing: false,
  // Fourth additive ABI v1 increment: Z-Image Union 2.x inpaint. It reuses
  // the control image above plus the shared brush mask canvas (white =
  // repaint, black = keep); no second image component is introduced.
  inpaintAvailable: true,
  drawing: false,
  lastPoint: null,
  // Model weight downloads: /api/models is a snapshot polled while transfers
  // are active. Per-file variant overrides are kept out of the rebuilt DOM so
  // a progress refresh never resets an operator's quantisation choice.
  models: null,
  globalVariant: "int8",
  modelVariants: {},
  modelGroupsOpen: {},
  manualGuide: null,
  configPreview: null,
};

// modelFamily already covers the new mode: every non-qwen21 model (zimage,
// zimage-control and zimage-inpaint) is a Z-Image family request, so
// checkpoint / 16-px steps still apply and qwen21-only fields stay hidden.
const modelFamily = () => ($("model").value.startsWith("qwen21") ? "qwen21" : "zimage");
const isEditMode = () => $("model").value === "qwen21-edit";
const isControlMode = () => $("model").value === "zimage-control";
const isInpaintMode = () => $("model").value === "zimage-inpaint";
// Inpaint is a strict superset of ControlNet: it always carries a control
// image, so the control upload / type / scale fields are shared by both.
const isControlLike = () => isControlMode() || isInpaintMode();

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
      // /api/preprocess advertises its operators and which WanGP ONNX
      // estimators are actually installed. It works without a native engine.
      state.preprocessCapabilities = h.preprocessing || null;
      syncPreprocessAvailability();
      // Editing is additive in ABI v1: an older native library simply has no
      // qzi_generate_qwen21_edit export. Do not offer a mode that would fail.
      state.editingAvailable = h.supports_image_editing === true;
      const editOption = document.querySelector('#model option[value="qwen21-edit"]');
      if (editOption) editOption.disabled = !state.editingAvailable;
      if (!state.editingAvailable && isEditMode()) {
        $("model").value = "qwen21";
        switchModel();
      }
      // Advanced editing is a further optional increment: disable its controls
      // rather than queueing a job that the library cannot serve. Basic editing
      // remains available on libs that only export the second increment.
      state.advancedEditingAvailable = h.supports_advanced_editing === true;
      syncAdvancedControls();
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
      // Inpaint is the fourth additive increment: an older library has no
      // qzi_generate_zimage_inpaint export. Disable the option (and fall back
      // to plain zimage if it was selected) so no job is queued to fail.
      state.inpaintAvailable = h.supports_inpaint === true;
      const inpaintOption = document.querySelector('#model option[value="zimage-inpaint"]');
      if (inpaintOption) {
        inpaintOption.disabled = !state.inpaintAvailable;
        inpaintOption.title = state.inpaintAvailable
          ? "Z-Image Union 2.x inpaint: control image + brush mask"
          : "Unavailable: the loaded native library has no Z-Image inpaint export";
      }
      if (!state.inpaintAvailable && isInpaintMode()) {
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
  const inpaint = isInpaintMode();
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
  // image, type and scale controls. "zimage-inpaint" reuses that control
  // block plus the shared brush mask canvas; qwen21-only sub-fields of the
  // edit block are hidden via their own data-only="qwen21-edit" markers.
  document.querySelectorAll("[data-only]").forEach((el) => {
    const only = el.getAttribute("data-only");
    el.hidden = only === "qwen21" ? family !== "qwen21"
      : only === "qwen21-edit" ? !edit
      : only === "zimage" ? family !== "zimage"
      : only === "zimage-control" ? !isControlLike()
      : only === "edit-or-inpaint" ? !(edit || inpaint)
      : only === "zimage-inpaint" ? !inpaint
      : false;
  });

  const isBase = family === "qwen21" && $("tier").value === "base";
  document.querySelectorAll("[data-requires='base']").forEach((el) => {
    el.querySelectorAll("input, select, textarea").forEach((input) => { input.disabled = !isBase; });
  });
  $("negative").disabled = !isBase;
  syncMaskControls();
  syncControlScaleHint();
  syncAdvancedControls();
  syncPreprocessDefaults();
  syncPreprocessSideHint();
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

/* The native C ABI consumes a pre-formatted planar RGB control raster. The
 * WebUI — not the NuGet library — builds that raster: an ordinary photo is
 * re-encoded to 8-bit PNG by a canvas, POSTed to /api/preprocess, and the
 * returned guide is what /api/jobs receives. WanGP's pose / canny / depth /
 * scribble guide choices are therefore available without adding any imaging
 * dependency to the core package. */

function syncPreprocessDefaults() {
  if (state.preprocessTouched) return;
  const select = $("control-preprocess");
  if (!select) return;
  // Inpaint needs the photograph itself as its canvas (WanGP "V" / raw).
  // Standalone ControlNet defaults to a canny guide so an upload is never
  // silently consumed "as-is" any more.
  select.value = isInpaintMode() ? "raw" : "canny";
}

function syncPreprocessAvailability() {
  const learned = state.preprocessCapabilities?.learned;
  const pose = document.querySelector('#control-preprocess option[value="pose"]');
  if (pose) pose.title = learned?.learned_pose === true
    ? "WanGP YOLOX-L + DWPose ONNX assets found"
    : "Requires QZI_PREPROCESS_DIR with pose/yolox_l.onnx and pose/dw-ll_ucoco_384.onnx";
  const depth = document.querySelector('#control-preprocess option[value="depth"]');
  if (depth) depth.title = learned?.learned_depth === true
    ? "Depth Anything V2 ONNX model found"
    : "Uses the built-in deterministic relative-depth proxy (no ONNX model configured)";
}

function syncPreprocessSideHint() {
  $("control-preprocess-side-hint").textContent = `${$("control-preprocess-side").value} px`;
}

function setControlInfo(text) {
  const el = $("control-preprocess-info");
  if (!el) return;
  el.textContent = text || "";
  el.hidden = !text;
}

function renderControlPreview() {
  $("control-preview").src = state.controlPng;
  $("control-empty").hidden = true;
  $("control-editor").hidden = false;
  // Inpaint reuses the qwen21-edit brush mask canvas verbatim: the guide is
  // its backdrop and sets the mask resolution. Only reset the painted mask
  // when that resolution actually changed (e.g. re-running with a new
  // max-side), so changing canny -> scribble does not erase brush work.
  if (isInpaintMode()) {
    const mask = $("mask-canvas");
    if (mask.width !== state.controlWidth || mask.height !== state.controlHeight) {
      mask.width = state.controlWidth;
      mask.height = state.controlHeight;
      clearMask();
    }
    $("reference-preview").src = state.controlPng;
    $("mask-editor").hidden = false;
  }
}

async function runPreprocess() {
  if (!state.controlSourcePng || state.preprocessing) return;
  const mode = $("control-preprocess").value;
  const button = $("control-reprocess");
  state.preprocessing = true;
  if (button) button.disabled = true;
  setControlInfo(`Building ${mode} control guide…`);
  try {
    const result = await api("/api/preprocess", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: state.controlSourcePng,
        mode,
        control_type: $("control-type").value,
        max_side: Number($("control-preprocess-side").value),
      }),
    });
    state.controlPng = result.control_image;
    state.controlWidth = result.width;
    state.controlHeight = result.height;
    state.controlMeta = result;
    // Keep the descriptive control_type in step with the guide actually
    // produced (posing a canny map as "auto" would mislead the Union
    // weights). "raw" has no dedicated canonical type, so an explicit
    // "auto" selection is left alone — that is the inpaint default.
    if (result.control_type && !(result.control_type === "raw" && $("control-type").value === "auto"))
      $("control-type").value = result.control_type;
    renderControlPreview();
    const provider = result.learned ? `${result.provider} (ONNX)` : result.provider;
    setControlInfo(`Guide: ${result.mode} · ${result.width}×${result.height} · ${provider}` +
      (result.warning ? ` — ${result.warning}` : ""));
    showToast(`Control guide ready (${result.width}×${result.height}) · ${result.mode} via ${provider}.`);
  } catch (error) {
    // A failed estimator must never leave the form unusable. Fall back to
    // the raw upload; the API error for `pose` already names the missing
    // asset directory, so the operator knows exactly what to configure.
    state.controlPng = state.controlSourcePng;
    state.controlWidth = state.controlSourceWidth;
    state.controlHeight = state.controlSourceHeight;
    state.controlMeta = null;
    $("control-type").value = "raw";
    renderControlPreview();
    const message = String(error.message || error);
    setControlInfo(`Preprocessing failed: ${message}. Using the uploaded image as a raw guide.`);
    showToast(message);
  } finally {
    state.preprocessing = false;
    if (button) button.disabled = false;
  }
}

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
    state.controlSourcePng = canvas.toDataURL("image/png");
    state.controlSourceWidth = width;
    state.controlSourceHeight = height;
    state.controlPng = state.controlSourcePng;
    state.controlWidth = width;
    state.controlHeight = height;
    renderControlPreview();
    await runPreprocess();
  } catch (error) {
    showToast(`Could not read the control image: ${error.message || error}`);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function removeControlImage() {
  state.controlSourcePng = null;
  state.controlSourceWidth = 0;
  state.controlSourceHeight = 0;
  state.controlPng = null;
  state.controlWidth = 0;
  state.controlHeight = 0;
  state.controlMeta = null;
  $("control-file").value = "";
  $("control-preview").removeAttribute("src");
  $("control-editor").hidden = true;
  $("control-empty").hidden = false;
  setControlInfo(null);
  if (isInpaintMode()) {
    clearMask();
    $("reference-preview").removeAttribute("src");
    $("mask-editor").hidden = true;
  }
}

function syncControlScaleHint() {
  const slider = $("control-scale");
  if (!slider) return;
  const value = Number(slider.value);
  $("control-scale-hint").textContent = value === 0 ? "0 (engine default 0.65)" : value.toFixed(2);
}

// --------------------------------------------------------------------- form

// Advanced editing is additive: when the loaded library lacks the third
// increment, disable its controls and keep ordinary editing usable.
function syncAdvancedControls() {
  const available = state.advancedEditingAvailable;
  document.querySelectorAll("#advanced-edit input, #advanced-edit select").forEach((el) => {
    el.disabled = !available;
  });
  $("advanced-unavailable").hidden = available;
}

// Only send the advanced envelope when at least one knob differs from the
// default. Otherwise the request stays a plain qwen21-edit job (second
// increment), which works on older native libraries too.
function advancedEditFields() {
  if (!state.advancedEditingAvailable) return null;
  const margin = (id) => Number($(id).value) || 0;
  const top = margin("margin-top");
  const bottom = margin("margin-bottom");
  const left = margin("margin-left");
  const right = margin("margin-right");
  const mode = Number($("model-mode").value) || 0;
  const rgba = Number($("rgba").value) || 0;
  if (top === 0 && bottom === 0 && left === 0 && right === 0 && mode === 0 && rgba === 0) {
    return null;
  }
  return {
    margin_top: top,
    margin_bottom: bottom,
    margin_left: left,
    margin_right: right,
    model_mode: mode,
    rgba,
  };
}

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
    const advanced = advancedEditFields();
    if (advanced) Object.assign(body, advanced);
  }
  if (isControlLike()) {
    body.control_image = state.controlPng;
    body.control_type = $("control-type").value;
    body.control_scale = Number($("control-scale").value);
    // Inpaint carries the same control fields plus the single-channel brush
    // mask (white = repaint, black = keep). submit() guarantees a painted
    // mask so the mapper never sees a missing mask_image on this path.
    if (isInpaintMode()) {
      body.mask_image = maskPng();
    }
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
  if (isControlLike() && !state.controlPng) {
    showToast("Upload a control image before queueing a ControlNet / inpaint job.");
    $("control-file").focus();
    return;
  }
  if (isControlLike() && state.preprocessing) {
    showToast("Control guide preprocessing is still running — please wait a moment.");
    return;
  }
  if (isInpaintMode() && !maskPng()) {
    showToast("Paint a mask (white = repaint) before queueing an inpaint job.");
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
    const advancedBadge = job.advanced_edit
      ? `<span class="job-edit-badge">adv${job.outpaint ? " · outpaint" : ""}${job.rgba ? " · rgba" : ""}</span>`
      : "";
    // Inpaint is a superset of ControlNet: show one inpaint badge (including
    // the has_mask flag) instead of a redundant separate control badge.
    const controlBadge = job.control && !job.inpaint
      ? `<span class="job-control-badge">control</span>`
      : "";
    const inpaintBadge = job.inpaint
      ? `<span class="job-edit-badge">inpaint${job.has_mask ? " · mask" : ""}</span>`
      : "";
    const actions = [];
    if (!terminal) actions.push(`<button class="btn btn-ghost btn-sm" data-action="cancel" data-id="${job.id}">Cancel</button>`);
    else actions.push(`<button class="btn btn-ghost btn-sm" data-action="retry" data-id="${job.id}">Retry</button>`);
    return `<tr>
      <td>#${job.id}${job.retry_of ? `<div class="hint">↻ #${job.retry_of}</div>` : ""}</td>
      <td>${escapeHtml(job.model)}${editBadge}${advancedBadge}${controlBadge}${inpaintBadge}<div class="job-prompt" title="${escapeHtml(job.prompt)}">${escapeHtml(job.prompt)}</div></td>
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

// ------------------------------------------------------- model downloads

/* The weight catalogue is served by /api/models. It is entirely read-only
 * until the operator presses a button: the browser sends file ids / variant
 * ids / control actions only, never URLs or destination paths. Progress is
 * polled instead of streamed so a refresh, a proxy or a reconnect can never
 * lose a transfer (the server keeps .qzi-part files across restarts). */

function formatBytes(value) {
  const n = Number(value) || 0;
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${n} B`;
}

function formatSpeed(bps) {
  const n = Number(bps) || 0;
  return n > 0 ? `${(n / 1024 ** 2).toFixed(1)} MB/s` : "—";
}

function copyText(text, label) {
  if (!text) { showToast(`Nothing to copy for ${label}.`); return; }
  const fallback = () => {
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch { /* unsupported */ }
    area.remove();
    showToast(ok ? `${label} copied.` : `Could not copy ${label}; select the text manually.`);
  };
  if (navigator.clipboard?.writeText) {
    navigator.clipboard.writeText(text)
      .then(() => showToast(`${label} copied.`))
      .catch(fallback);
  } else fallback();
}

function findModelFile(id) {
  for (const group of state.models?.groups || []) {
    const found = group.files.find((f) => f.id === id);
    if (found) return found;
  }
  return null;
}

function selectedVariant(file) {
  if (!file?.variants?.length) return null;
  const wanted = state.modelVariants[file.id] || state.globalVariant;
  return file.variants.find((v) => v.id === wanted)
      || file.variants.find((v) => v.recommended)
      || file.variants[0];
}

function modelsActive(snapshot) {
  if (!snapshot) return false;
  if (snapshot.paused || snapshot.queue_length > 0) return true;
  return snapshot.groups.some((g) => g.files.some((f) =>
    ["downloading", "queued", "paused"].includes(f.status)));
}

async function refreshModels(showErrors = true) {
  try {
    state.models = await api("/api/models");
    renderModels();
  } catch (error) {
    $("models-pill").className = "pill pill-bad";
    $("models-pill").textContent = "● /api/models unavailable";
    if (showErrors) showToast(String(error.message || error));
  }
}

function renderModels() {
  const snapshot = state.models;
  if (!snapshot) return;
  // Rebuilding the group DOM every 1.2 s is fine for progress, but must not
  // steal focus from an open <select> or a checkbox the operator is using.
  const focused = document.activeElement;
  const container = $("models-groups");
  if (focused && container.contains(focused) && /^(SELECT|INPUT|TEXTAREA)$/.test(focused.tagName)) return;

  const allFiles = snapshot.groups.flatMap((g) => g.files);
  const installed = allFiles.filter((f) => f.status === "installed").length;
  const downloading = allFiles.filter((f) => f.status === "downloading").length;
  const pill = $("models-pill");
  pill.className = snapshot.paused ? "pill pill-wait"
    : downloading ? "pill pill-ok" : "pill pill-ok";
  pill.textContent = snapshot.paused ? "● paused"
    : downloading ? `● downloading ${downloading}`
    : `● ${installed}/${allFiles.length} installed`;
  $("models-summary").textContent =
    `${formatBytes(snapshot.installed_bytes)} / ${formatBytes(snapshot.total_bytes)} · ` +
    `mirror ${snapshot.mirror || "none"} · queue ${snapshot.queue_length}`;
  $("config-target-path").textContent = snapshot.config_target_path || "QwenZImage.config.json";
  $("config-platform-name").textContent = snapshot.config_platform || "platform";
  $("models-pause").disabled = snapshot.paused === true;
  $("models-resume").disabled = snapshot.paused !== true;

  container.innerHTML = snapshot.groups.map((group) => {
    const open = state.modelGroupsOpen[group.id] !== false;
    const groupInstalled = group.files.filter((f) => f.status === "installed").length;
    return `<details class="model-group" data-group="${escapeHtml(group.id)}" ${open ? "open" : ""}>
      <summary data-group="${escapeHtml(group.id)}">
        ${escapeHtml(group.name)}
        <span class="model-tag ${group.optional ? "optional" : "required"}">${group.optional ? "optional" : "core"}</span>
        <span class="model-group-head-meta">${groupInstalled}/${group.files.length} installed · ${escapeHtml(group.config_key)}</span>
      </summary>
      <div class="model-group-body">
        <p class="model-group-path">
          <code>${escapeHtml(group.config_key)}</code> → <code>${escapeHtml(group.resolved_path)}</code>
          · ${escapeHtml(group.path_source)}${group.configured ? "" : " · <strong>not configured</strong>"}
          · env <code>${escapeHtml(group.environment_variable)}</code>
        </p>
        <p class="model-file-desc">${escapeHtml(group.description)}</p>
        ${group.note ? `<p class="model-file-note">${escapeHtml(group.note)}</p>` : ""}
        ${group.files.map((file) => renderModelFile(file)).join("")}
      </div>
    </details>`;
  }).join("");
}

function renderModelFile(file) {
  const variant = selectedVariant(file);
  const total = Number(file.total_bytes) || 0;
  const done = Number(file.downloaded_bytes) || 0;
  const pct = total > 0 ? Math.min(100, Math.max(0, (done / total) * 100)) : 0;
  const showProgress = ["downloading", "queued", "paused", "partial"].includes(file.status);
  const busy = ["downloading", "queued"].includes(file.status);
  return `<div class="model-file" data-file="${escapeHtml(file.id)}">
    <div class="model-file-top">
      <span class="model-file-name">${escapeHtml(file.name)}</span>
      <span class="model-tag ${file.required ? "required" : "optional"}">${file.required ? "required" : "optional"}</span>
      <span class="model-status ${escapeHtml(file.status)}">${escapeHtml(file.status)}</span>
      <span class="hint">${escapeHtml(file.id)} · ${formatBytes(file.installed_bytes)} installed</span>
      <div class="model-file-actions" style="margin-left:auto">
        <select data-model-action="variant" data-file="${escapeHtml(file.id)}" aria-label="Variant for ${escapeHtml(file.name)}">
          ${(file.variants || []).map((v) => `<option value="${escapeHtml(v.id)}" ${v.id === variant?.id ? "selected" : ""}>${escapeHtml(v.label)}</option>`).join("")}
        </select>
        <button class="btn btn-primary btn-sm" data-model-action="download" data-file="${escapeHtml(file.id)}" ${busy ? "disabled" : ""} type="button">Download</button>
        <button class="btn btn-ghost btn-sm" data-model-action="reset" data-file="${escapeHtml(file.id)}" type="button">Reset</button>
      </div>
    </div>
    <div class="model-file-desc">${escapeHtml(file.description)}</div>
    ${file.note ? `<div class="model-file-note">${escapeHtml(file.note)}</div>` : ""}
    ${file.error ? `<div class="model-error">${escapeHtml(file.error)}</div>` : ""}
    ${showProgress ? `<div class="model-progress">
      <div class="progress-track"><div class="progress-fill" style="width:${pct.toFixed(1)}%"></div></div>
      <div class="model-progress-text">${formatBytes(done)} / ${formatBytes(total)} (${pct.toFixed(1)}%) · resume from ${formatBytes(file.resume_from)} · ${formatSpeed(file.speed_bps)}</div>
    </div>` : ""}
    <div class="model-target">Target: <code>${escapeHtml(file.target_path)}</code>
      <button class="btn btn-ghost btn-sm" data-model-action="copy-path" data-file="${escapeHtml(file.id)}" type="button">Copy path</button>
    </div>
    ${(file.variants || []).map((v) => `<div class="model-variant-row">
      <span class="model-variant-label">${escapeHtml(v.label)}${v.recommended ? " ★" : ""}</span>
      <a class="model-url" href="${escapeHtml(v.url)}" target="_blank" rel="noopener">${escapeHtml(v.url)}</a>
      <button class="btn btn-ghost btn-sm" data-model-action="copy-url" data-file="${escapeHtml(file.id)}" data-variant="${escapeHtml(v.id)}" type="button">Copy URL</button>
      ${v.source_url && v.source_url !== v.url ? `<div class="model-source-url">WanGP source: <code>${escapeHtml(v.source_url)}</code></div>` : ""}
    </div>`).join("")}
    ${variant ? `<pre class="models-pre model-command">${escapeHtml(variant.curl_command)}</pre>
      <div class="model-file-actions">
        <button class="btn btn-ghost btn-sm" data-model-action="copy-curl" data-file="${escapeHtml(file.id)}" type="button">Copy resumable curl command</button>
        <span class="hint">${formatBytes(variant.size)} → <code>${escapeHtml(variant.target_path)}</code></span>
      </div>` : ""}
  </div>`;
}

function downloadPayload(extra = {}) {
  return Object.assign({
    variant: $("models-variant-select").value,
    force: $("models-force").checked,
  }, extra);
}

async function postModelsDownload(payload) {
  try {
    const result = await api("/api/models/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    showToast(`Queued ${result.queued} download(s)` +
      (result.skipped?.length ? ` · skipped ${result.skipped.length} already present` : "") +
      (result.note ? ` · ${result.note}` : ""));
    await refreshModels(false);
  } catch (error) { showToast(String(error.message || error)); }
}

async function postModelsControl(payload) {
  try {
    const result = await api("/api/models/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    showToast(`${result.action}: ${result.reset} partial(s) cleared · queue ${result.queue_length}` +
      (result.paused ? " · paused" : ""));
    await refreshModels(false);
  } catch (error) { showToast(String(error.message || error)); }
}

async function onModelsAction(event) {
  const el = event.target.closest("[data-model-action]");
  if (!el || el.tagName === "SELECT") return;
  const action = el.dataset.modelAction;
  const fileId = el.dataset.file;
  const file = findModelFile(fileId);
  if (!file) return;
  const variant = selectedVariant(file);
  if (action === "download") {
    await postModelsDownload(downloadPayload({ files: [fileId], variant: variant?.id }));
  } else if (action === "reset") {
    await postModelsControl({ action: "reset", file: fileId });
  } else if (action === "copy-path") {
    copyText(file.target_path, "target path");
  } else if (action === "copy-url") {
    const v = file.variants.find((x) => x.id === el.dataset.variant);
    copyText(v?.url, "download URL");
  } else if (action === "copy-curl") {
    copyText(variant?.curl_command, "resumable curl command");
  }
}

function onModelsVariantChange(event) {
  const el = event.target.closest("select[data-model-action='variant']");
  if (!el) return;
  state.modelVariants[el.dataset.file] = el.value;
  renderModels();
}

async function loadManualGuide() {
  try {
    const response = await fetch("/api/models/manual");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.manualGuide = await response.text();
    $("manual-content").textContent = state.manualGuide;
    showToast("Manual download guide loaded.");
  } catch (error) { showToast(String(error.message || error)); }
}

function downloadManualGuide() {
  const text = state.manualGuide;
  if (!text) { showToast("Load the guide first."); return; }
  const url = URL.createObjectURL(new Blob([text], { type: "text/markdown;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "QwenZImage-models.md";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function renderConfigStatus(result) {
  const overrides = (result.active_overrides || []);
  $("config-status").textContent = result.message +
    (overrides.length ? ` Active overrides: ${overrides.join("; ")}.` : "") +
    (result.backup_path ? ` Backup: ${result.backup_path}.` : "") +
    " Restart the WebUI to apply.";
}

async function previewConfig() {
  try {
    state.configPreview = await api("/api/models/config");
    $("config-content").textContent = state.configPreview.merged_json || "";
    renderConfigStatus(state.configPreview);
  } catch (error) { showToast(String(error.message || error)); }
}

async function applyConfig() {
  const target = state.models?.config_target_path || "QwenZImage.config.json";
  if (!confirm(`Merge the six resolved model directories into the ${state.models?.config_platform || "platform"} section of\n${target}?\n\nA timestamped backup is made first, and the WebUI must be restarted afterwards.`)) return;
  try {
    const result = await api("/api/models/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ apply: true }),
    });
    state.configPreview = result;
    $("config-content").textContent = result.merged_json || "";
    renderConfigStatus(result);
    showToast(`Config updated (backup: ${result.backup_path || "new file"}). Restart the WebUI.`);
    await refreshModels(false);
  } catch (error) { showToast(String(error.message || error)); }
}

// Fast polling while a transfer is active, slow polling otherwise. Skipping
// hidden tabs keeps a long download from burning CPU in a background window.
async function modelsPollLoop() {
  if (!document.hidden) await refreshModels(false);
  setTimeout(modelsPollLoop, modelsActive(state.models) ? 1200 : 12000);
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
  ["margin-top", "margin-bottom", "margin-left", "margin-right",
   "model-mode", "rgba"].forEach((id) => {
    $(id).addEventListener("input", markCustom);
    $(id).addEventListener("change", markCustom);
  });

  // Z-Image ControlNet Union: upload, guide preprocessing, removal and scale.
  $("control-file").addEventListener("change", (event) => loadControlImage(event.target.files?.[0]));
  $("control-remove").addEventListener("click", removeControlImage);
  $("control-reprocess").addEventListener("click", runPreprocess);
  $("control-preprocess").addEventListener("change", () => {
    state.preprocessTouched = true;
    if (state.controlSourcePng) runPreprocess();
  });
  $("control-preprocess-side").addEventListener("input", syncPreprocessSideHint);
  $("control-preprocess-side").addEventListener("change", () => {
    syncPreprocessSideHint();
    if (state.controlSourcePng) runPreprocess();
  });
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

  // Model weights: downloads, resumable transfer controls, manual addresses
  // and the config-file merge. Every action goes through a snake_case API
  // that only accepts file ids / variant ids / control verbs.
  $("models-refresh").addEventListener("click", () => refreshModels());
  $("models-download-required").addEventListener("click", () =>
    postModelsDownload(downloadPayload({ files: [] })));
  $("models-download-all").addEventListener("click", () =>
    postModelsDownload(downloadPayload({ all: true })));
  $("models-pause").addEventListener("click", () => postModelsControl({ action: "pause" }));
  $("models-resume").addEventListener("click", () => postModelsControl({ action: "resume" }));
  $("models-reset-all").addEventListener("click", () => {
    if (!confirm("Discard all .qzi-part / .qzi-etag temporary files? Completed weights are untouched.")) return;
    postModelsControl({ action: "reset-all" });
  });
  $("models-variant-select").addEventListener("change", (event) => {
    state.globalVariant = event.target.value;
    state.modelVariants = {};
    renderModels();
  });
  $("models-groups").addEventListener("click", (event) => {
    const summary = event.target.closest("summary[data-group]");
    if (summary) {
      const groupId = summary.dataset.group;
      // The browser toggles <details> after this event, so read it next tick.
      setTimeout(() => {
        const details = [...$("models-groups").querySelectorAll("details.model-group")]
          .find((d) => d.dataset.group === groupId);
        if (details) state.modelGroupsOpen[groupId] = details.open;
      }, 0);
    }
    onModelsAction(event);
  });
  $("models-groups").addEventListener("change", onModelsVariantChange);
  $("manual-load").addEventListener("click", loadManualGuide);
  $("manual-download").addEventListener("click", downloadManualGuide);
  $("manual-copy").addEventListener("click", () => copyText(state.manualGuide, "manual guide"));
  $("config-preview").addEventListener("click", previewConfig);
  $("config-copy").addEventListener("click", () =>
    copyText(state.configPreview?.merged_json, "merged config JSON"));
  $("config-apply").addEventListener("click", applyConfig);
}

async function boot() {
  wire();
  syncModelUI();
  await Promise.all([refreshHealth(), refreshPresets(), refreshJobs(),
                     refreshOutputs(), refreshModels(false)]);
  connectEvents();
  modelsPollLoop();
  // Safety net for browsers/proxies that buffer SSE: reflect job state anyway.
  setInterval(() => {
    const busy = [...state.jobs.values()].some((j) => j.state === "queued" || j.state === "running");
    if (busy) refreshJobs();
  }, 3000);
  setInterval(refreshHealth, 30000);
}

boot();
