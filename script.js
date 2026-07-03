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

const MAX_SOURCES = 16; // 每帧最多注入的波源数量（V形尾波需要更多槽位）

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
  gesture: "none",       // "index" | "palm" | "none"
  flowerCooldown: 0,     // 下次可以生花的时间戳（ms）
  palmX: 0,
  palmY: 0,
  // 新增：手指停留检测
  lastMoveTime: 0,
  hoverX: 0,
  hoverY: 0,
  isHovering: false,
  // 新增：速度轨迹效果
  lastSpeed: 0,
  trailCooldown: 0,
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

// 正在绽放的花朵队列（每朵花分 3 个阶段注入，营造"绽开"动画感）
const activeFlowers = [];

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
    // 系数降到 0.22：波速更慢，涟漪范围更小更subtle
    float laplacian = (hl + hr + hu + hd) - 4.0 * h;
    vel += laplacian * 0.22;

    // 注入手指/指针带来的扰动
    // src.w 可为负值（手指划过处产生凹陷，两侧被推高）
    for (int i = 0; i < ${MAX_SOURCES}; i++) {
      if (i >= u_sourceCount) break;
      vec4 src = u_sources[i];
      vec2 d = uv - src.xy;
      d.x *= u_aspect;
      float r = max(src.z, 0.0006);
      float falloff = exp(-dot(d, d) / (r * r));
      vel += falloff * src.w;
    }

    // 速度阻尼：0.982 快速衰减，让涟漪更克制
    vel *= 0.9820;
    h += vel;
    // 高度快速回落
    h *= 0.9965;

    // 边缘软衰减：靠近四边时压低能量，得到柔和反射而非生硬弹回
    vec2 edge = min(uv, 1.0 - uv);
    float border = smoothstep(0.0, 0.055, min(edge.x, edge.y));
    h   *= mix(0.88, 1.0, border);
    vel *= mix(0.88, 1.0, border);

    gl_FragColor = vec4(h, vel, 0.0, 1.0);
  }
`;

// 渲染 pass：用高度场法线折射摄像头画面 + 色散 + 多层焦散 + 细腻光影
const RENDER_FRAGMENT = `
  precision highp float;

  uniform sampler2D u_sim;
  uniform sampler2D u_video;
  uniform vec2 u_texel;
  uniform vec2 u_resolution;
  uniform vec2 u_videoSize;
  uniform float u_time;
  varying vec2 v_uv;

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
    float h  = texture2D(u_sim, uv).r;

    float hlu = texture2D(u_sim, uv + vec2(-u_texel.x, u_texel.y)).r;
    float hru = texture2D(u_sim, uv + vec2(u_texel.x, u_texel.y)).r;
    float hld = texture2D(u_sim, uv + vec2(-u_texel.x, -u_texel.y)).r;
    float hrd = texture2D(u_sim, uv + vec2(u_texel.x, -u_texel.y)).r;

    vec2 grad = vec2(hl - hr, hd - hu);
    grad += vec2((hlu + hld) - (hru + hrd), (hlu + hru) - (hld + hrd)) * 0.5;
    vec3 normal = normalize(vec3(grad * 14.0, 1.0));

    float gradLen = length(grad);
    float activity = clamp(gradLen * 18.0, 0.0, 1.0);

    float curvature = (hl + hr + hu + hd - 4.0 * h) * 0.5;
    float curvatureIntensity = clamp(abs(curvature) * 100.0, 0.0, 1.0);

    float refrStr = 0.012 + activity * 0.010;
    vec2 refractOffset = normal.xy * refrStr;

    vec2 base = coverUv(uv);
    float disperseScale = 0.15;
    vec2 rOffset = refractOffset * (1.0 + disperseScale * 0.008);
    vec2 gOffset = refractOffset;
    vec2 bOffset = refractOffset * (1.0 - disperseScale * 0.008);

    float r = texture2D(u_video, clamp(base + rOffset, 0.001, 0.999)).r;
    float g = texture2D(u_video, clamp(base + gOffset, 0.001, 0.999)).g;
    float b = texture2D(u_video, clamp(base + bOffset, 0.001, 0.999)).b;
    vec3 color = vec3(r, g, b);

    if (activity > 0.5) {
      vec2 refracted2 = coverUv(uv + refractOffset * 1.15 + vec2(0.0005, -0.0004));
      vec3 refractC = texture2D(u_video, clamp(refracted2, 0.001, 0.999)).rgb;
      color = mix(color, refractC, clamp(activity * 0.18, 0.0, 0.18));
    }

    vec3 L1 = normalize(vec3( 0.42,  0.58, 0.92));
    vec3 L2 = normalize(vec3(-0.48,  0.28, 0.86));

    float spec1 = pow(max(dot(normal, L1), 0.0), 38.0) * 0.6;
    float spec2 = pow(max(dot(normal, L2), 0.0), 24.0) * 0.25;

    float fresnel = pow(1.0 - clamp(normal.z, 0.0, 1.0), 1.8);

    vec3 specColor = vec3(0.95, 0.93, 0.90);
    color += specColor * (spec1 + spec2) * (0.35 + fresnel * 0.30);

    float crest = clamp(h * 9.0, 0.0, 1.0);
    float gradAngle = atan(grad.y, grad.x + 0.0001);

    float caustic1 = pow(max(0.0,
      sin(uv.x * 24.0 - uv.y * 10.0 + u_time * 0.48 + gradAngle * 2.5)
    ), 18.0) * 0.5;

    float caustic2 = pow(max(0.0,
      sin(uv.x * 17.0 + uv.y * 20.0 - u_time * 0.38 - gradAngle * 1.4)
    ), 22.0) * 0.3;

    float causticTotal = caustic1 + caustic2;

    vec3 causticColor = vec3(0.93, 0.90, 0.85);
    color += causticColor * causticTotal * crest * 0.08;

    color += vec3(0.025, 0.024, 0.023) * crest * 0.25;
    float trough = clamp(-h * 7.0, 0.0, 1.0);
    color *= 1.0 - trough * 0.04;

    float edgeGlow = curvatureIntensity * activity * 0.05;
    color += vec3(0.88, 0.86, 0.84) * edgeGlow;

    color = mix(color, color * vec3(1.015, 1.00, 0.99), activity * 0.12);
    color += vec3(0.006, 0.006, 0.005);

    float shimmer1 = pow(max(0.0,
      sin(uv.x * 15.0 + uv.y * 13.0 + u_time * 0.38)),
      30.0) * 0.008;

    float shimmer2 = pow(max(0.0,
      sin(uv.x * 32.0 - uv.y * 28.0 + u_time * 0.68)),
      40.0) * 0.005;

    color += vec3(0.96, 0.95, 0.92) * (shimmer1 + shimmer2);

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

  // 重建模拟网格：分辨率从 256 提升到 320，让细波纹有足够精度
  const aspect = width / height;
  simH = 320;
  simW = clamp(Math.round(simH * aspect), 120, 640);

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
    hint.textContent = "食指划过水面 · 五指张开唤出水花";
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
    if (!landmarks) {
      handState.gesture = "none";
      return;
    }

    // 检测有多少根手指伸出（通过指尖与关节的相对位置判断）
    const isThumbOut = Math.abs(landmarks[4].x - landmarks[2].x) > 0.08;
    const isIndexOut = landmarks[8].y < landmarks[6].y - 0.05;
    const isMiddleOut = landmarks[12].y < landmarks[10].y - 0.05;
    const isRingOut = landmarks[16].y < landmarks[14].y - 0.05;
    const isPinkyOut = landmarks[20].y < landmarks[18].y - 0.05;
    const extendedCount = [isThumbOut, isIndexOut, isMiddleOut, isRingOut, isPinkyOut]
      .filter(Boolean).length;

    // 判断手势类型
    const now = performance.now();
    if (extendedCount >= 4) {
      // 张开手掌：生成花朵图案
      handState.gesture = "palm";
      const palmX = (1 - landmarks[9].x) * width;   // 中指根部作为掌心
      const palmY = landmarks[9].y * height;
      handState.palmX = palmX;
      handState.palmY = palmY;

      if (now >= handState.flowerCooldown) {
        spawnFlower(palmX, palmY);
        handState.flowerCooldown = now + 650;  // 650ms 冷却，避免过于频繁
      }
    } else if (isIndexOut && extendedCount <= 2) {
      // 单指模式：食指划水
      handState.gesture = "index";
      const tipX = (1 - landmarks[8].x) * width;
      const tipY = landmarks[8].y * height;

      if (!handState.hasSmoothPoint) {
        handState.smoothX = tipX;
        handState.smoothY = tipY;
        handState.hasSmoothPoint = true;
      }
      handState.smoothX += (tipX - handState.smoothX) * 0.62;
      handState.smoothY += (tipY - handState.smoothY) * 0.62;
      feedPoint(handState.smoothX, handState.smoothY, 1);
    } else {
      handState.gesture = "none";
    }

    handState.ready = true;
    handState.seenAt = now;
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

// 根据移动速度，注入极窄的高密度单线波源——使水面保留清晰的手指轨迹纹路
// 物理逻辑：
//   • 极小注入半径（0.004~0.008）+ 高密度插值 = 路径上形成连续细线扰动
//   • 较慢的波速（laplacian 0.28）确保波纹不会迅速向外扩散稀释
//   • 结果：可以清楚看出一根手指划过的具体路径和方向
function feedPoint(x, y, confidence) {
  const now = performance.now();
  if (!pointer.seen) {
    pointer.x = x;
    pointer.y = y;
    pointer.seen = true;
    pointer.movedAt = now;
    handState.lastMoveTime = now;
    return;
  }
  pointer.px = pointer.x;
  pointer.py = pointer.y;
  pointer.x = x;
  pointer.y = y;
  pointer.movedAt = now;

  const dx = x - pointer.px;
  const dy = y - pointer.py;
  const speed = Math.hypot(dx, dy);

  if (speed < 0.1) return;

  // 每 3.5 像素注入一个波源，确保轨迹连续无断点
  const steps = Math.min(Math.max(Math.round(speed / 3.5), 1), MAX_SOURCES - 1);

  for (let s = 0; s < steps; s++) {
    const t  = (s + 0.5) / steps;
    const ix = pointer.px + dx * t;
    const iy = pointer.py + dy * t;

    // 极窄半径：在模拟网格上对应 1.5~2 个像素，轨迹非常纤细
    const radius   = clamp(0.004 + speed * 0.00010, 0.004, 0.008);
    // 强度随速度增强：划得快，波纹越深
    const strength = clamp(0.085 + speed * 0.0030, 0.085, 0.22) * confidence;

    pushSource(ix, iy, radius, strength);
  }
}

// ---------------------------------------------------------------------------
// 花朵涟漪：五指张开时在掌心生成水波小花
// 设计理念：清晰的花瓣边界 + 水的流动感 + 真实物理 + 视觉美感
// ---------------------------------------------------------------------------

// 将一朵花加入队列（cx/cy 为掌心屏幕像素坐标）
function spawnFlower(cx, cy) {
  const numPetals = 5; // 固定5片花瓣，像参考图
  const pSize = Math.min(width, height) * (0.10 + Math.random() * 0.04);

  activeFlowers.push({
    cx,
    cy,
    baseAngle: Math.random() * Math.PI * 2,
    pSize,
    numPetals,
    phase: 0,
  });
}

// 每帧调用：生成像参考图那样清晰的5瓣水花
function tickFlowers() {
  for (let i = activeFlowers.length - 1; i >= 0; i--) {
    const f = activeFlowers[i];
    const t = f.phase;

    // ═══════════════════════════════════════════════════════════════════════
    // 阶段 0: 花心（小圆形凹陷）
    // ═══════════════════════════════════════════════════════════════════════
    if (t === 0) {
      pushSource(f.cx, f.cy, 0.032, -0.28);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 阶段 1-2: 花心外围（形成圆润的中心）
    // ═══════════════════════════════════════════════════════════════════════
    else if (t === 1) {
      const numCenter = 6;
      for (let k = 0; k < numCenter; k++) {
        const a = (k / numCenter) * Math.PI * 2;
        const r = f.pSize * 0.12;
        pushSource(
          f.cx + Math.cos(a) * r,
          f.cy + Math.sin(a) * r,
          0.018, -0.16
        );
      }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 阶段 2-6: 5片花瓣主体形成（清晰的边界线）
    // ═══════════════════════════════════════════════════════════════════════
    else if (t >= 2 && t <= 6) {
      const progress = (t - 2) / 4; // 0 到 1

      for (let k = 0; k < f.numPetals; k++) {
        const a = f.baseAngle + (k / f.numPetals) * Math.PI * 2;

        // 花瓣长度逐渐增长
        const rMax = f.pSize * (0.35 + progress * 0.50);

        // ═══ 花瓣中心线（暗色主体）═══
        const numSpine = 4;
        for (let j = 0; j < numSpine; j++) {
          const spineT = j / (numSpine - 1);
          const r = rMax * spineT;

          pushSource(
            f.cx + Math.cos(a) * r,
            f.cy + Math.sin(a) * r,
            0.016,
            -0.12 * (1.0 - spineT * 0.3)
          );
        }

        // ═══ 花瓣边缘（亮边轮廓）═══
        if (t >= 3) {
          const edgeProgress = Math.min((t - 3) / 3, 1.0);
          const perpAngle = a + Math.PI * 0.5;

          // 沿花瓣长度方向的边缘点
          const numEdge = 5;
          for (let j = 0; j < numEdge; j++) {
            const edgeT = j / (numEdge - 1);
            const r = rMax * edgeT * edgeProgress;

            // 花瓣宽度：根部宽，尖端窄
            const width = f.pSize * 0.18 * Math.sin(edgeT * Math.PI * 0.65);

            // 左右边界
            for (let side of [-1, 1]) {
              pushSource(
                f.cx + Math.cos(a) * r + Math.cos(perpAngle) * width * side,
                f.cy + Math.sin(a) * r + Math.sin(perpAngle) * width * side,
                0.012,
                0.14
              );
            }
          }
        }
      }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 阶段 7-9: 花瓣尖端高光
    // ═══════════════════════════════════════════════════════════════════════
    else if (t >= 7 && t <= 9) {
      for (let k = 0; k < f.numPetals; k++) {
        const a = f.baseAngle + (k / f.numPetals) * Math.PI * 2;
        const r = f.pSize * 0.88;

        pushSource(
          f.cx + Math.cos(a) * r,
          f.cy + Math.sin(a) * r,
          0.011,
          0.18
        );
      }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 阶段 10-12: 外围柔和扩散
    // ═══════════════════════════════════════════════════════════════════════
    else if (t >= 10 && t <= 12) {
      const ringProgress = (t - 10) / 2;
      const numRing = f.numPetals * 3;
      const ringR = f.pSize * (1.10 + ringProgress * 0.25);

      for (let k = 0; k < numRing; k++) {
        const a = (k / numRing) * Math.PI * 2;
        pushSource(
          f.cx + Math.cos(a) * ringR,
          f.cy + Math.sin(a) * ringR,
          0.012,
          0.05 - ringProgress * 0.02
        );
      }
    }

    f.phase++;
    if (f.phase > 14) activeFlowers.splice(i, 1);
  }
}

function bindInput() {
  // 安静的指针兜底：主要用于桌面调试 / 摄像头不可用时
  window.addEventListener("pointermove", (e) => feedPoint(e.clientX, e.clientY, 0.9));
  window.addEventListener("pointerdown", (e) => {
    // 点触入水：先注入一个 feedPoint 跟新逻辑同步，再叠加一个扩散圆环模拟"入水一点"
    feedPoint(e.clientX, e.clientY, 1);
    pushSource(e.clientX, e.clientY, 0.032, 0.14);
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
  const recentlySeen = handState.ready && performance.now() - handState.seenAt < 1400;
  if (recentlySeen) {
    hint.style.opacity = "0";
  } else if (handState.detector) {
    hint.style.opacity = "1";
    hint.textContent = "食指划过水面 · 五指张开唤出水花";
  }
}

function animate(time) {
  scanHand(time);
  updateHint(time);

  const hasVideo = uploadVideoFrame();

  // 每帧先推进花朵绽放动画，再跑两步物理模拟
  tickFlowers();
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
