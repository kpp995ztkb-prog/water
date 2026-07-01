/**
 * 一池野春水 — 摄像头 + 食指手势驱动的真实水面模拟
 *
 * 与旧版「掌心花粉」最大的不同：水波不再是 shader 里若干预设圆环的解析叠加，
 * 而是一套真正的 GPU 高度场水面模拟（双缓冲 framebuffer ping-pong + 波动方程）。
 * 手指划过处向高度场注入扰动，波纹会自然向外扩散、遇边软反射、彼此干涉、缓缓衰减。
 * 渲染时用高度场的法线对实时摄像头画面做折射，并叠加焦散高光与冷暖微光。
 */

const video = document.querySelector("#camera");
const canvas = document.querySelector("#water");
const hint = document.querySelector(".hint");

const gl = canvas.getContext("webgl", {
  alpha: false,
  antialias: false,
  depth: false,
  stencil: false,
  preserveDrawingBuffer: false,
});

const MAX_SOURCES = 8; // 每帧最多注入的波源数量

// ---- 运行状态 ----
const pointer = {
  x: 0,
  y: 0,
  px: 0,
  py: 0,
  seen: false,
  movedAt: 0,
};

const handState = {
  detector: null,
  scanning: false,
  smoothX: 0,
  smoothY: 0,
  hasSmoothPoint: false,
  seenAt: 0,
  ready: false,
};

let width = 0;
let height = 0;
let dpr = 1;

// 模拟网格（低于屏幕分辨率，物理在这上面跑）
let simW = 0;
let simH = 0;

let lastTime = performance.now();
let lastHandScan = 0;
let lastHintUpdate = 0;
let videoFrameReady = false;

// 本帧待注入的波源：{ x, y (sim uv 0..1), radius, strength }
const sources = [];

// WebGL 资源
let simProgram = null;
let renderProgram = null;
let quadBuffer = null;
let videoTexture = null;
let simTextures = [];
let simFramebuffers = [];
let simRead = 0; // ping-pong 索引
let simType = null;
let simFilter = null;
let simLoc = {};
let renderLoc = {};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

// ---------------------------------------------------------------------------
// Shader 编译
// ---------------------------------------------------------------------------
function createShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const error = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(error);
  }
  return shader;
}

function createProgram(vertexSource, fragmentSource) {
  const vs = createShader(gl.VERTEX_SHADER, vertexSource);
  const fs = createShader(gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const error = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(error);
  }
  return program;
}

const QUAD_VERTEX = `
  attribute vec2 a_position;
  varying vec2 v_uv;
  void main() {
    v_uv = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
  }
`;

// 水面模拟 pass：波动方程 + 波源注入 + 边缘软衰减
// 纹理通道：R = 高度，G = 速度
const SIM_FRAGMENT = `
  precision highp float;

  uniform sampler2D u_prev;
  uniform vec2 u_texel;        // 1 / simSize
  uniform float u_aspect;      // simW / simH，用于让波源在屏幕上呈圆形
  uniform int u_sourceCount;
  uniform vec4 u_sources[${MAX_SOURCES}]; // xy=uv, z=radius, w=strength
  varying vec2 v_uv;

  void main() {
    vec2 uv = v_uv;
    vec4 self = texture2D(u_prev, uv);
    float h = self.r;
    float vel = self.g;

    float hl = texture2D(u_prev, uv - vec2(u_texel.x, 0.0)).r;
    float hr = texture2D(u_prev, uv + vec2(u_texel.x, 0.0)).r;
    float hu = texture2D(u_prev, uv + vec2(0.0, u_texel.y)).r;
    float hd = texture2D(u_prev, uv - vec2(0.0, u_texel.y)).r;

    // 离散拉普拉斯 → 加速度
    float laplacian = (hl + hr + hu + hd) - 4.0 * h;
    vel += laplacian * 0.48;

    // 注入手指/指针带来的扰动
    for (int i = 0; i < ${MAX_SOURCES}; i++) {
      if (i >= u_sourceCount) break;
      vec4 src = u_sources[i];
      vec2 d = uv - src.xy;
      d.x *= u_aspect;
      float r = max(src.z, 0.0008);
      float falloff = exp(-dot(d, d) / (r * r));
      vel += falloff * src.w;
    }

    vel *= 0.9965;   // 速度阻尼，让水面缓缓平息
    h += vel;
    h *= 0.9992;     // 高度回落，避免能量长期堆积

    // 边缘软衰减：靠近四边时压低能量，得到柔和而非生硬的反射
    vec2 edge = min(uv, 1.0 - uv);
    float border = smoothstep(0.0, 0.045, min(edge.x, edge.y));
    h *= mix(0.92, 1.0, border);
    vel *= mix(0.92, 1.0, border);

    gl_FragColor = vec4(h, vel, 0.0, 1.0);
  }
`;

// 渲染 pass：用高度场法线折射摄像头画面 + 焦散高光 + 冷暖微光
const RENDER_FRAGMENT = `
  precision highp float;

  uniform sampler2D u_sim;
  uniform sampler2D u_video;
  uniform vec2 u_texel;
  uniform vec2 u_resolution;
  uniform vec2 u_videoSize;
  uniform float u_time;
  varying vec2 v_uv;

  // cover 映射 + 水平镜像，使摄像头铺满且符合“照镜子”的方向
  vec2 coverUv(vec2 screenUv) {
    vec2 ratio = vec2(
      max((u_resolution.x / u_resolution.y) / (u_videoSize.x / u_videoSize.y), 1.0),
      max((u_resolution.y / u_resolution.x) / (u_videoSize.y / u_videoSize.x), 1.0)
    );
    vec2 uv = (screenUv - 0.5) * ratio + 0.5;
    uv.x = 1.0 - uv.x;
    return uv;
  }

  void main() {
    vec2 uv = v_uv;

    float hl = texture2D(u_sim, uv - vec2(u_texel.x, 0.0)).r;
    float hr = texture2D(u_sim, uv + vec2(u_texel.x, 0.0)).r;
    float hu = texture2D(u_sim, uv + vec2(0.0, u_texel.y)).r;
    float hd = texture2D(u_sim, uv - vec2(0.0, u_texel.y)).r;
    float h = texture2D(u_sim, uv).r;

    // 由高度梯度求水面法线
    vec2 grad = vec2(hl - hr, hd - hu);
    vec3 normal = normalize(vec3(grad * 9.0, 1.0));

    float activity = clamp(length(grad) * 12.0, 0.0, 1.0);

    // 折射：沿法线水平分量偏移采样坐标
    vec2 refractOffset = normal.xy * 0.42;
    vec2 base = coverUv(uv);
    vec2 soft = coverUv(uv + refractOffset * 0.018 + vec2(0.0012, -0.0008));
    vec3 color = texture2D(u_video, clamp(base + refractOffset * 0.02, 0.001, 0.999)).rgb;
    vec3 softened = texture2D(u_video, clamp(soft, 0.001, 0.999)).rgb;
    color = mix(color, softened, clamp(activity * 0.5, 0.0, 0.4)); // 起伏处更柔，似雨雾

    // 焦散 / 镜面高光
    vec3 lightDir = normalize(vec3(0.35, 0.55, 0.9));
    float spec = pow(max(dot(normal, lightDir), 0.0), 28.0);
    float crest = clamp(h * 6.0, 0.0, 1.0);
    color += vec3(0.85, 0.95, 0.92) * spec * 0.5;
    color += vec3(0.6, 0.8, 0.78) * crest * 0.06;

    // 冷暖微光：暗部偏青，活跃处透出一点暖意
    color = mix(color, color * vec3(0.92, 1.01, 1.04), 0.5);            // 整体偏青绿
    color = mix(color, color * vec3(1.06, 1.0, 0.9), activity * 0.25);   // 涟漪带暖
    color += vec3(0.012, 0.014, 0.013);

    // 极轻的水面波光，避免画面太“数码”
    float sheen =
      pow(max(0.0, sin(uv.x * 14.0 + uv.y * 11.0 + u_time * 0.3)), 20.0) * 0.012;
    color += vec3(sheen);

    gl_FragColor = vec4(color, 1.0);
  }
`;

// ---------------------------------------------------------------------------
// 浮点渲染目标探测（高度场需要高精度，byte 纹理精度不够）
// ---------------------------------------------------------------------------
function pickSimTextureType() {
  gl.getExtension("OES_texture_float");
  const halfExt = gl.getExtension("OES_texture_half_float");
  const floatLinear = gl.getExtension("OES_texture_float_linear");
  const halfLinear = gl.getExtension("OES_texture_half_float_linear");
  gl.getExtension("WEBGL_color_buffer_float");
  gl.getExtension("EXT_color_buffer_half_float");

  const candidates = [
    { type: gl.FLOAT, linear: !!floatLinear },
  ];
  if (halfExt) {
    candidates.push({ type: halfExt.HALF_FLOAT_OES, linear: !!halfLinear });
  }

  for (const candidate of candidates) {
    const filter = candidate.linear ? gl.LINEAR : gl.NEAREST;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 4, 4, 0, gl.RGBA, candidate.type, null);

    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fb);
    gl.deleteTexture(tex);

    if (status === gl.FRAMEBUFFER_COMPLETE) {
      return { type: candidate.type, filter };
    }
  }
  return null;
}

function createSimTarget(w, h) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, simFilter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, simFilter);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, simType, null);

  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { tex, fb };
}

function clearSimTargets() {
  for (const fb of simFramebuffers) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

// ---------------------------------------------------------------------------
// 初始化
// ---------------------------------------------------------------------------
function setupWebgl() {
  if (!gl) {
    hint.textContent = "这个浏览器不支持 WebGL，换 Chrome 再试试";
    return false;
  }

  const sim = pickSimTextureType();
  if (!sim) {
    hint.textContent = "这台设备不支持浮点水面模拟，换新一点的浏览器试试";
    return false;
  }
  simType = sim.type;
  simFilter = sim.filter;

  simProgram = createProgram(QUAD_VERTEX, SIM_FRAGMENT);
  renderProgram = createProgram(QUAD_VERTEX, RENDER_FRAGMENT);

  quadBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1, 1, -1, -1, 1,
    -1, 1, 1, -1, 1, 1,
  ]), gl.STATIC_DRAW);

  simLoc = {
    position: gl.getAttribLocation(simProgram, "a_position"),
    prev: gl.getUniformLocation(simProgram, "u_prev"),
    texel: gl.getUniformLocation(simProgram, "u_texel"),
    aspect: gl.getUniformLocation(simProgram, "u_aspect"),
    sourceCount: gl.getUniformLocation(simProgram, "u_sourceCount"),
    sources: gl.getUniformLocation(simProgram, "u_sources"),
  };
  renderLoc = {
    position: gl.getAttribLocation(renderProgram, "a_position"),
    sim: gl.getUniformLocation(renderProgram, "u_sim"),
    video: gl.getUniformLocation(renderProgram, "u_video"),
    texel: gl.getUniformLocation(renderProgram, "u_texel"),
    resolution: gl.getUniformLocation(renderProgram, "u_resolution"),
    videoSize: gl.getUniformLocation(renderProgram, "u_videoSize"),
    time: gl.getUniformLocation(renderProgram, "u_time"),
  };

  videoTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, videoTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);

  return true;
}

function resize() {
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  width = window.innerWidth;
  height = window.innerHeight;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  // 重建模拟网格：固定高度，宽度按屏幕比例，保持近似方形的网格单元
  const aspect = width / height;
  simH = 256;
  simW = clamp(Math.round(simH * aspect), 96, 512);

  // 释放旧目标
  for (const t of simTextures) gl.deleteTexture(t);
  for (const f of simFramebuffers) gl.deleteFramebuffer(f);

  const a = createSimTarget(simW, simH);
  const b = createSimTarget(simW, simH);
  simTextures = [a.tex, b.tex];
  simFramebuffers = [a.fb, b.fb];
  simRead = 0;
  clearSimTargets();
}

// ---------------------------------------------------------------------------
// 摄像头 + 手势
// ---------------------------------------------------------------------------
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    await setupHandTracking();
  } catch {
    hint.textContent = "请允许摄像头权限后刷新页面";
  }
}

async function setupHandTracking() {
  hint.textContent = "正在加载手势模型，也可以先用鼠标拂过水面";
  try {
    const vision = await import(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs"
    );
    const fileset = await vision.FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
    );
    handState.detector = await vision.HandLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
      },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.45,
      minHandPresenceConfidence: 0.45,
      minTrackingConfidence: 0.45,
    });
    hint.textContent = "伸出食指，轻轻拂过这池春水";
  } catch {
    hint.textContent = "手势模型加载失败，先用鼠标拂过水面试试";
  }
}

async function scanHand(time) {
  if (!handState.detector || handState.scanning || video.readyState < 2 || time - lastHandScan < 60) {
    return;
  }
  lastHandScan = time;
  handState.scanning = true;
  try {
    const results = handState.detector.detectForVideo(video, time);
    const landmarks = results.landmarks?.[0];
    if (!landmarks) return;

    const indexTip = landmarks[8];
    // 显示是镜像的，所以横坐标取反
    const tipX = (1 - indexTip.x) * width;
    const tipY = indexTip.y * height;

    if (!handState.hasSmoothPoint) {
      handState.smoothX = tipX;
      handState.smoothY = tipY;
      handState.hasSmoothPoint = true;
    }
    handState.smoothX += (tipX - handState.smoothX) * 0.5;
    handState.smoothY += (tipY - handState.smoothY) * 0.5;

    handState.ready = true;
    handState.seenAt = performance.now();
    feedPoint(handState.smoothX, handState.smoothY, 1);
  } catch {
    handState.detector = null;
    hint.textContent = "手势识别运行失败，先用鼠标拂过水面试试";
  } finally {
    handState.scanning = false;
  }
}

// 向水面注入一个波源（输入为屏幕像素坐标）
function pushSource(x, y, radius, strength) {
  if (sources.length >= MAX_SOURCES) return;
  sources.push({
    x: clamp(x / width, 0, 1),
    y: clamp(1 - y / height, 0, 1), // 纹理 Y 向上
    radius,
    strength,
  });
}

// 根据移动速度，把一个点转化为连续的拂水扰动
function feedPoint(x, y, confidence) {
  const now = performance.now();
  if (!pointer.seen) {
    pointer.x = x;
    pointer.y = y;
    pointer.seen = true;
  }
  pointer.px = pointer.x;
  pointer.py = pointer.y;
  pointer.x = x;
  pointer.y = y;
  pointer.movedAt = now;

  const speed = Math.hypot(x - pointer.px, y - pointer.py);
  if (speed > 0.4) {
    const strength = clamp(0.015 + speed * 0.0012, 0.015, 0.09) * confidence;
    pushSource(x, y, 0.012, strength);
  }
}

function bindInput() {
  // 安静的指针兜底：主要用于桌面调试 / 摄像头不可用时
  window.addEventListener("pointermove", (e) => feedPoint(e.clientX, e.clientY, 0.9));
  window.addEventListener("pointerdown", (e) => {
    feedPoint(e.clientX, e.clientY, 1);
    pushSource(e.clientX, e.clientY, 0.02, 0.12);
  });
}

// ---------------------------------------------------------------------------
// 渲染循环
// ---------------------------------------------------------------------------
function drawQuad(attribLoc) {
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
  gl.enableVertexAttribArray(attribLoc);
  gl.vertexAttribPointer(attribLoc, 2, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}

function stepSimulation(injectThisStep) {
  const writeIndex = 1 - simRead;
  gl.bindFramebuffer(gl.FRAMEBUFFER, simFramebuffers[writeIndex]);
  gl.viewport(0, 0, simW, simH);
  gl.useProgram(simProgram);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, simTextures[simRead]);
  gl.uniform1i(simLoc.prev, 0);
  gl.uniform2f(simLoc.texel, 1 / simW, 1 / simH);
  gl.uniform1f(simLoc.aspect, simW / simH);

  const count = injectThisStep ? sources.length : 0;
  gl.uniform1i(simLoc.sourceCount, count);
  if (count > 0) {
    const data = new Float32Array(MAX_SOURCES * 4);
    for (let i = 0; i < count; i++) {
      data[i * 4] = sources[i].x;
      data[i * 4 + 1] = sources[i].y;
      data[i * 4 + 2] = sources[i].radius;
      data[i * 4 + 3] = sources[i].strength;
    }
    gl.uniform4fv(simLoc.sources, data);
  }

  drawQuad(simLoc.position);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  simRead = writeIndex;
}

function renderToScreen(time) {
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.useProgram(renderProgram);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, simTextures[simRead]);
  gl.uniform1i(renderLoc.sim, 0);

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, videoTexture);
  gl.uniform1i(renderLoc.video, 1);

  gl.uniform2f(renderLoc.texel, 1 / simW, 1 / simH);
  gl.uniform2f(renderLoc.resolution, canvas.width, canvas.height);
  gl.uniform2f(renderLoc.videoSize, video.videoWidth || 1280, video.videoHeight || 720);
  gl.uniform1f(renderLoc.time, time / 1000);

  drawQuad(renderLoc.position);
}

function uploadVideoFrame() {
  if (!videoFrameReady || !video.videoWidth || video.readyState < 2) return false;
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, videoTexture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
  return true;
}

function updateHint(time) {
  if (time - lastHintUpdate < 600) return;
  lastHintUpdate = time;
  if (handState.ready && performance.now() - handState.seenAt < 1400) {
    hint.style.opacity = "0";
  } else if (handState.detector) {
    hint.style.opacity = "1";
    hint.textContent = "伸出食指，轻轻拂过这池春水";
  }
}

function animate(time) {
  scanHand(time);
  updateHint(time);

  const hasVideo = uploadVideoFrame();

  // 每帧跑两步模拟：更稳、波纹更顺滑；只在第一步注入波源
  stepSimulation(true);
  stepSimulation(false);
  sources.length = 0;

  if (hasVideo) {
    renderToScreen(time);
  } else {
    // 摄像头还没就绪：用冷青底色占位
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.08, 0.11, 0.12, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  lastTime = time;
  requestAnimationFrame(animate);
}

// ---------------------------------------------------------------------------
// 启动
// ---------------------------------------------------------------------------
if (setupWebgl()) {
  resize();
  bindInput();
  video.addEventListener("playing", () => { videoFrameReady = true; });
  video.addEventListener("timeupdate", () => { videoFrameReady = true; });
  window.addEventListener("resize", resize);
  startCamera();
  requestAnimationFrame(animate);

  window.waterDebug = {
    get simSize() { return [simW, simH]; },
    get simType() { return simType === gl.FLOAT ? "FLOAT" : "HALF_FLOAT"; },
    get videoReady() { return videoFrameReady; },
  };
}
