
'use strict';

// ==================== 元素 ====================
const mic = document.getElementById('mic');
const phoneBtn = document.getElementById('phone');
const hint = document.getElementById('hint');
const logEl = document.getElementById('log');
const logToggle = document.getElementById('logToggle');
const unreadEl = document.getElementById('unread');
const health = document.getElementById('health');
const typeIn = document.getElementById('typeIn');
const recTimer = document.getElementById('recTimer');
const orb = document.getElementById('orb');
const wave = document.getElementById('wave');
const meterEl = document.getElementById('meter');
const meterFill = document.getElementById('meterFill');
const devInfo = document.getElementById('devInfo');
const devSel = document.getElementById('dev');
const devRow = document.querySelector('.devRow');
const devRefresh = document.getElementById('devRefresh');
const diagEl = document.getElementById('diag');
const buildStamp = document.getElementById('buildStamp');

const octx = orb.getContext('2d');
const wctx = wave.getContext('2d');

// ==================== HUD 遥测（纯展示，不参与任何逻辑）====================
const dotEl      = document.getElementById('dot');
const stateLblEl = document.getElementById('stateLbl');
const tState     = document.getElementById('tState');
const tDev       = document.getElementById('tDev');
const tLevel     = document.getElementById('tLevel');
const tLevelFill = document.querySelector('#tLevelBar i');
const tVad       = document.getElementById('tVad');
const tClock     = document.getElementById('tClock');
const tMode      = document.getElementById('tMode');
const BOOT_AT    = Date.now();
const STATE_COL  = { idle: 'var(--cy)', recording: 'var(--red)',
                     thinking: 'var(--violet)', speaking: 'var(--grn)' };

function paintTelemetry() {
  const s = state;
  dotEl.className = s === 'recording' ? 'rec'
                  : s === 'thinking'  ? 'think'
                  : s === 'speaking'  ? 'speak' : 'on';
  stateLblEl.textContent = s;
  tState.textContent = s;
  tState.style.color = STATE_COL[s] || 'var(--cy)';
  tDev.textContent = devLabel || '—';
  let pct = 0;
  try { pct = Math.round(Math.min(1, level()) * 100); } catch (e) { pct = 0; }
  tLevel.textContent = pct + '%';
  if (tLevelFill) tLevelFill.style.width = pct + '%';
  tVad.textContent  = phoneOn ? 'armed' : 'offline';
  tMode.textContent = phoneOn ? 'full-duplex' : 'half-duplex';
  const sec = Math.floor((Date.now() - BOOT_AT) / 1000);
  tClock.textContent = String(Math.floor(sec / 60)).padStart(2, '0')
                     + ':' + String(sec % 60).padStart(2, '0');
}
setInterval(paintTelemetry, 200);

// ==================== 状态 ====================
let state = 'idle';          // idle | recording | thinking | speaking
let recorder = null;
let chunks = [];
let recStart = 0;
let recTick = null;
let ripples = [];
let logOpen = false;      // 记录默认隐藏
let unread = 0;           // 隐藏期间收到的条数
let devLabel = '未知设备';
let silentSince = 0;
let silenceWarned = false;
let inputDeviceId = '';
let micTrack = null;
let diagAt = 0;

// ---- 电话模式（连续对话）----
let phoneOn = false;
let phoneStream = null;
let phoneRec = null;
let phoneChunks = [];
let phoneTurnAt = 0;
let hasSpeech = false;
let speechFrom = 0;
let silenceFrom = 0;
let bargeFrom = 0;
let micAnalyser = null;
let curAudio = null;

const hasMic = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
function micWhy() {
  if (location.protocol === 'file:') return '页面用文件方式打开了（file://）';
  if (!window.isSecureContext) return location.host + ' 不是安全上下文（非 localhost 的 http 会被浏览器禁用麦克风）';
  if (!navigator.mediaDevices) return '浏览器不支持 MediaDevices（请用 Chrome / Edge）';
  return '';
}

// ==================== 音频分析 ====================
let audioCtx = null, analyser = null, freqBuf = null, timeBuf = null, micSrc = null;

function ensureCtx() {
  if (!audioCtx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return null;
    audioCtx = new AC();
  }
  if (audioCtx.state === 'suspended') audioCtx.resume();
  return audioCtx;
}

function makeAnalyser() {
  const ctx = ensureCtx();
  if (!ctx) return null;
  const an = ctx.createAnalyser();
  an.fftSize = 1024;
  an.smoothingTimeConstant = 0.75;
  if (freqBuf === null || freqBuf.length !== an.frequencyBinCount) {
    freqBuf = new Uint8Array(an.frequencyBinCount);
  }
  if (timeBuf === null || timeBuf.length !== an.fftSize) {
    timeBuf = new Uint8Array(an.fftSize);
  }
  return an;
}

// 麦克风接分析器（不接 destination，避免自己听自己）
let audioGraphErr = '';
function attachMic(stream) {
  const ctx = ensureCtx();
  const an = makeAnalyser();
  if (!ctx || !an) { audioGraphErr = '浏览器不支持 AudioContext'; return false; }
  try {
    micSrc = ctx.createMediaStreamSource(stream);
    micSrc.connect(an);
    analyser = an;
    micAnalyser = an;
    audioGraphErr = '';
    return true;
  } catch (e) {
    analyser = null;
    micAnalyser = null;
    audioGraphErr = e.message || String(e);
    return false;
  }
}

// 只拆麦克风侧（电话模式会一直持有它）
function detachMic() {
  try { if (micSrc) micSrc.disconnect(); } catch (e) {}
  micSrc = null;
  micAnalyser = null;
  micTrack = null;
  if (analyser !== playAnalyser) analyser = playAnalyser;
}

// ==================== 流式播放器 ====================
// 为什么不用 <audio> 元素播：TTS 现在是"一句一段"地回来的，每段一个独立 mp3。
// 用 <audio> 逐段播，段间会有 100~300ms 的重新缓冲黑屏感；
// Web Audio 的 BufferSource 能按 ctx.currentTime **精确排期**，段与段首尾相接。
const player = { ctx: null, an: null, gain: null, srcs: [], nextAt: 0, playing: false };
let playAnalyser = null;

function playerReady() {
  const ctx = ensureCtx();
  if (!ctx) return null;
  if (!player.ctx) {
    player.ctx = ctx;
    player.an = makeAnalyser();
    player.gain = ctx.createGain();
    player.gain.connect(player.an);
    player.an.connect(ctx.destination);
    // 浏览器会把 AudioContext 挂起（自动播放策略）。挂起时 BufferSource 不走时间轴，
    // onended 永远不触发 —— 界面就会卡在"说话中"。这里兜住，别让状态机僵死。
    ctx.onstatechange = () => {
      if (ctx.state !== 'running' && player.playing) {
        stopPlayback();
        if (turn && turn.done) finishTurn(turn);
        else setHint('音频被浏览器挂起了 —— 点一下页面任意处继续');
      }
    };
  }
  if (player.ctx.state === 'suspended') { player.ctx.resume().catch(() => {}); }
  return player;
}

// 停止播放（打断 / 挂断 / 新一轮）。注意：不清 gain 节点，下一个 buffer 直接接上
function stopPlayback() {
  for (const s of player.srcs) {
    try { s.stop(); } catch (e) {}
    try { s.disconnect(); } catch (e) {}
  }
  player.srcs = [];
  player.nextAt = 0;
  player.playing = false;
  playAnalyser = null;
  analyser = micAnalyser;
  curAudio = null;
}

// 兼容旧调用点（原来拆 <audio> 用的）
function detachPlay() { stopPlayback(); }

// 兼容老调用：整条拆掉
function detach() { detachPlay(); detachMic(); }

// ==================== 流式音频入队 ====================
function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// 一段音频到了：解码 -> 按时间轴排期 -> 播。返回 Promise 不阻塞 SSE 读取
let pendingAudio = 0;     // 正在解码、还没排进队列的段数（防止 done 先到误判"播完了"）

async function enqueueAudio(b64, tok) {
  const p = playerReady();
  if (!p) return;
  pendingAudio += 1;
  let buf = null;
  try {
    buf = await p.ctx.decodeAudioData(b64ToBytes(b64).buffer);
  } catch (e) {
    console.warn('音频段解码失败，跳过', e);            // 单段失败不影响后面
  } finally {
    pendingAudio -= 1;
  }
  if (buf && isToken(tok)) {
    const src = p.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(p.gain);
    // 排期：比"现在"晚一点点（防止 start 时间已过被吞），并与上一段首尾相接
    const at = Math.max(p.ctx.currentTime + 0.04, p.nextAt);
    src.start(at);
    p.nextAt = at + buf.duration + 0.01;
    p.srcs.push(src);
    p.playing = true;
    playAnalyser = p.an;
    analyser = p.an;                                  // 让球体跟着她的声音跳
    if (state !== 'speaking') setState('speaking');
    src.onended = () => {
      p.srcs = p.srcs.filter(s => s !== src);
      if (!p.srcs.length) { p.playing = false; onSpeechDrained(); }
    };
    return;
  }
  // 这一段丢了：如果这是最后一段，别让界面卡在"说话中"
  if (pendingAudio === 0 && !p.srcs.length) onSpeechDrained();
}

// 给某个 analyser 算音量
function analyserLevel(an) {
  if (!an) return 0;
  if (!timeBuf || timeBuf.length !== an.fftSize) timeBuf = new Uint8Array(an.fftSize);
  an.getByteTimeDomainData(timeBuf);
  let sum = 0;
  for (let i = 0; i < timeBuf.length; i++) {
    const v = (timeBuf[i] - 128) / 128;
    sum += v * v;
  }
  return Math.min(1, Math.sqrt(sum / timeBuf.length) * 4.5);
}

// 音量（0~1）：时域 RMS，放大后裁剪
function level() {
  if (!analyser || !timeBuf) return 0;
  analyser.getByteTimeDomainData(timeBuf);
  let sum = 0;
  for (let i = 0; i < timeBuf.length; i++) {
    const v = (timeBuf[i] - 128) / 128;
    sum += v * v;
  }
  return Math.min(1, Math.sqrt(sum / timeBuf.length) * 4.5);
}

function spectrum() {
  if (!analyser || !freqBuf) return null;
  analyser.getByteFrequencyData(freqBuf);
  return freqBuf;
}

// ==================== 颜色 ====================
const PAL = {
  // 弧反应堆的三个特征：高饱和青、白热芯、暗底衬托。低饱和会掉成「灰塑料」
  idle:      { h: 191, s: 96, l: 56 },   // 弧堆青（JARVIS 主色）
  recording: { h: 28,  s: 96, l: 58 },   // 琥珀（警示）
  thinking:  { h: 205, s: 88, l: 68 },   // 冷白蓝（思考：更亮更冷，不换色系）
  speaking:  { h: 176, s: 92, l: 54 },   // 青绿（发声）
};

// 尊重系统「减少动态效果」
const REDUCED = window.matchMedia
  ? window.matchMedia('(prefers-reduced-motion: reduce)').matches
  : false;
function hsla(h, s, l, a) {
  return 'hsla(' + h + ',' + Math.max(0, Math.min(100, s)) + '%,'
         + Math.max(0, Math.min(100, l)) + '%,' + a + ')';
}

// ==================== 画布尺寸 ====================
function fitCanvas(cv, ctx) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return { w: 0, h: 0 };
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w: w, h: h };
}

// ==================== 画球 ====================
function drawOrb(now, amp) {
  const { w, h } = fitCanvas(orb, octx);
  if (!w || !h) return;
  const cx = w / 2, cy = h / 2;
  const col = PAL[state] || PAL.idle;
  octx.clearRect(0, 0, w, h);

  const t = REDUCED ? now * 0.16 : now;          // 减少动效时几乎静止
  const base = Math.min(w, h) * 0.225;
  const breathe = 0.014 * Math.sin(t * 1.1);
  const kick = state === 'idle' ? 0 : amp * 0.12;
  const R = base * (1 + breathe + kick);
  const hot = state === 'idle' ? 0.35 : 0.55 + amp * 0.45;

  const spec = spectrum();

  // ---------------- 1) 外刻度盘（罗盘感）----------------
  const dialR = R * 1.66;
  for (let i = 0; i < 120; i++) {
    const a = (i / 120) * Math.PI * 2 - Math.PI / 2 + t * 0.045;
    const major = i % 10 === 0;
    const len = major ? R * 0.085 : R * 0.038;
    octx.beginPath();
    octx.strokeStyle = hsla(col.h, col.s, col.l + 12, major ? 0.55 : 0.14);
    octx.lineWidth = major ? 1.4 : 1;
    octx.moveTo(cx + Math.cos(a) * (dialR - len), cy + Math.sin(a) * (dialR - len));
    octx.lineTo(cx + Math.cos(a) * dialR, cy + Math.sin(a) * dialR);
    octx.stroke();
  }

  // ---------------- 2) 细环 + 反向旋转虚线环 ----------------
  octx.globalCompositeOperation = 'lighter';

  octx.beginPath();
  octx.strokeStyle = hsla(col.h, col.s, col.l, 0.14);
  octx.lineWidth = 1;
  octx.arc(cx, cy, R * 1.54, 0, Math.PI * 2);
  octx.stroke();

  octx.save();
  octx.translate(cx, cy);
  octx.rotate(-t * 0.12);
  octx.beginPath();
  octx.setLineDash([2, 9]);
  octx.strokeStyle = hsla(col.h, col.s, col.l + 6, 0.34);
  octx.lineWidth = 1;
  octx.arc(0, 0, R * 1.40, 0, Math.PI * 2);
  octx.stroke();
  octx.setLineDash([]);
  octx.restore();

  // ---------------- 3) HUD 弧段（3 组，缺口错开）----------------
  const groups = state === 'thinking' ? 4 : 3;
  for (let k = 0; k < groups; k++) {
    const a0 = t * (0.42 + k * 0.16) + k * (Math.PI * 2 / groups);
    const span = 0.62 - k * 0.06;
    octx.beginPath();
    octx.strokeStyle = hsla(col.h, col.s + 6, col.l + 14, 0.30 + 0.14 * Math.sin(t + k));
    octx.lineWidth = k === 0 ? 2.2 : 1.4;
    octx.arc(cx, cy, R * (1.28 + k * 0.065), a0, a0 + span);
    octx.stroke();
  }

  // ---------------- 4) 频谱环（64 段，随声音起伏）----------------
  const SEG = 64;
  for (let i = 0; i < SEG; i++) {
    const a = (i / SEG) * Math.PI * 2 - Math.PI / 2 + t * 0.16;
    let v;
    if (spec) {
      const pos = Math.pow(i / SEG, 1.5);
      const bin = Math.min(spec.length - 1, Math.floor(pos * spec.length * 0.6));
      v = spec[bin] / 255;
    } else {
      v = 0.16 + 0.12 * Math.sin(i * 0.45 + t * 2.0);   // 无音频时的巡游波
    }
    const len = R * (0.035 + v * 0.20);
    const r1 = R * 1.12;
    octx.beginPath();
    octx.strokeStyle = hsla(col.h, col.s, col.l + 16, 0.22 + v * 0.68);
    octx.lineWidth = 2;
    octx.moveTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
    octx.lineTo(cx + Math.cos(a) * (r1 + len), cy + Math.sin(a) * (r1 + len));
    octx.stroke();
  }

  // ---------------- 5) 粒子（贴着核心外层）----------------
  const N = 64;
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2 + t * 0.22;
    const wob = Math.sin(i * 0.8 + t * 1.9) * 0.012
              + amp * 0.05 * Math.sin(i * 1.7 + t * 6);
    const rr = R * (1.0 + wob);
    const sz = 0.9 + amp * 1.6 * (0.5 + 0.5 * Math.sin(i * 0.9 + t * 5));
    octx.beginPath();
    octx.fillStyle = hsla(col.h, col.s, col.l + 24, 0.35 + amp * 0.45);
    octx.arc(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr, sz, 0, Math.PI * 2);
    octx.fill();
  }

  // ---------------- 6) 波纹 ----------------
  if (!REDUCED && ((state === 'recording' && amp > 0.22) ||
                   (state === 'speaking' && Math.random() < 0.22))) {
    ripples.push({ r: R * 1.14, a: state === 'recording' ? 0.26 : 0.18 });
  }
  ripples = ripples.filter(rp => rp.a > 0.012);
  for (const rp of ripples) {
    rp.r += 1.5;
    rp.a *= 0.965;
    octx.beginPath();
    octx.strokeStyle = hsla(col.h, col.s, col.l + 8, rp.a);
    octx.lineWidth = 1;
    octx.arc(cx, cy, rp.r, 0, Math.PI * 2);
    octx.stroke();
  }

  octx.globalCompositeOperation = 'source-over';

  // ---------------- 7) 核心：暗底 + 自发光（弧反应堆）----------------
  // 7a) 球体本色：高饱和青，从瓷白一路压到深青黑。
  //     这层是"球"，不是"雾"—— 饱和度给足，边缘压暗，中间才不会灰成塑料
  const bodyGrad = octx.createRadialGradient(
    cx - R * 0.10, cy - R * 0.12, 0, cx, cy, R);
  bodyGrad.addColorStop(0.00, hsla(col.h, 100, 84, 0.97));
  bodyGrad.addColorStop(0.18, hsla(col.h, 100, 74, 0.95));
  bodyGrad.addColorStop(0.42, hsla(col.h, 100, 58, 0.93));
  bodyGrad.addColorStop(0.68, hsla(col.h + 2, 98, 38, 0.90));
  bodyGrad.addColorStop(0.88, hsla(col.h + 4, 96, 24, 0.82));
  bodyGrad.addColorStop(1.00, hsla(col.h + 6, 96, 16, 0.66));
  octx.beginPath();
  octx.fillStyle = bodyGrad;
  octx.arc(cx, cy, R, 0, Math.PI * 2);
  octx.fill();

  // ---------------- 8) 球内结构（裁剪在球内，画在光的下层）----------------
  octx.save();
  octx.beginPath();
  octx.arc(cx, cy, R - 0.5, 0, Math.PI * 2);
  octx.clip();

  // 8a) 经纬网 —— 绕竖轴自转的线框球
  const spin = t * 0.45;
  for (let m = 0; m < 3; m++) {
    const rx = Math.abs(Math.cos(spin + m * Math.PI / 3)) * R * 0.86;
    if (rx < 2) continue;
    octx.beginPath();
    octx.strokeStyle = hsla(col.h, 90, 78, 0.18);
    octx.lineWidth = 1;
    octx.ellipse(cx, cy, rx, R * 0.86, 0, 0, Math.PI * 2);
    octx.stroke();
  }
  for (const fy of [-0.45, 0, 0.45]) {
    const wr = R * 0.86 * Math.cos(Math.asin(fy));
    octx.beginPath();
    octx.strokeStyle = hsla(col.h, 90, 78, 0.13);
    octx.lineWidth = 1;
    octx.ellipse(cx, cy + R * 0.86 * fy, wr, wr * 0.22, 0, 0, Math.PI * 2);
    octx.stroke();
  }

  // 8b) 内部慢转虚线环
  octx.save();
  octx.translate(cx, cy);
  octx.rotate(t * 0.28);
  octx.beginPath();
  octx.setLineDash([1.5, 6]);
  octx.strokeStyle = hsla(col.h, 95, 82, 0.24);
  octx.lineWidth = 1;
  octx.arc(0, 0, R * 0.55, 0, Math.PI * 2);
  octx.stroke();
  octx.setLineDash([]);
  octx.restore();

  // 8c) 漂浮节点
  for (let i = 0; i < 5; i++) {
    const a = t * (0.18 + i * 0.05) + i * 2.4;
    const rr = R * (0.35 + 0.28 * Math.sin(i * 1.7 + t * 0.4));
    octx.beginPath();
    octx.fillStyle = hsla(col.h, 40, 96, 0.55 + amp * 0.30);
    octx.arc(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr * 0.9, 1.3, 0, Math.PI * 2);
    octx.fill();
  }

  // 8d) 扫描线
  const sy = cy - R + (((t * 0.22) % 1) * 2 * R);
  const sg = octx.createLinearGradient(0, sy - 16, 0, sy + 16);
  sg.addColorStop(0, hsla(col.h, 60, 92, 0));
  sg.addColorStop(0.5, hsla(col.h, 60, 92, 0.14 + amp * 0.16));
  sg.addColorStop(1, hsla(col.h, 60, 92, 0));
  octx.fillStyle = sg;
  octx.fillRect(cx - R, sy - 16, 2 * R, 32);
  octx.restore();   // 解除球内裁剪

  // ---------------- 9) 叠光：白热芯 + 外扩辉光 ----------------
  // 顺序很重要：先有"球体本色"（source-over），再往上加光（lighter）。
  // 反过来先加光再铺色，光会被颜色盖掉，就成了塑料球
  octx.globalCompositeOperation = 'lighter';

  const coreGrad = octx.createRadialGradient(
    cx - R * 0.05, cy - R * 0.06, 0, cx - R * 0.05, cy - R * 0.06, R * 0.46);
  coreGrad.addColorStop(0.00, hsla(col.h, 30, 100, 0.42 + amp * 0.16));
  coreGrad.addColorStop(0.32, hsla(col.h, 80, 94, 0.16));
  coreGrad.addColorStop(1.00, hsla(col.h, 80, 94, 0));
  octx.beginPath();
  octx.fillStyle = coreGrad;
  octx.arc(cx - R * 0.05, cy - R * 0.06, R * 0.46, 0, Math.PI * 2);
  octx.fill();

  const halo = octx.createRadialGradient(cx, cy, R * 0.96, cx, cy, R * 1.42);
  halo.addColorStop(0.00, hsla(col.h, 100, 62, 0.13 + amp * 0.10));
  halo.addColorStop(0.45, hsla(col.h, 100, 56, 0.045));
  halo.addColorStop(1.00, hsla(col.h, 100, 50, 0));
  octx.beginPath();
  octx.fillStyle = halo;
  octx.arc(cx, cy, R * 1.42, 0, Math.PI * 2);
  octx.fill();

  octx.globalCompositeOperation = 'source-over';

  // ---------------- 10) 边缘：亮环 + 内暗线，做出"壳" ----------------
  octx.beginPath();
  octx.strokeStyle = hsla(col.h, 100, 88, 0.52 + 0.34 * hot);
  octx.lineWidth = 1.2;
  octx.arc(cx, cy, R, 0, Math.PI * 2);
  octx.stroke();
  octx.beginPath();
  octx.strokeStyle = hsla(col.h + 4, 90, 14, 0.50);
  octx.lineWidth = 1;
  octx.arc(cx, cy, R - 2.5, 0, Math.PI * 2);
  octx.stroke();

  // 11) 反应堆内环 + 辐条
  const innerPulse = 0.5 + 0.5 * Math.sin(t * 1.6);
  octx.beginPath();
  octx.strokeStyle = hsla(col.h, col.s - 8, 82, 0.16 + innerPulse * 0.14);
  octx.lineWidth = 1.1;
  octx.arc(cx, cy, R * 0.30, 0, Math.PI * 2);
  octx.stroke();

  for (let k = 0; k < 3; k++) {
    const a = t * 0.5 + (k * Math.PI * 2) / 3;
    octx.beginPath();
    octx.strokeStyle = hsla(col.h, col.s, 80, 0.10 + amp * 0.20);
    octx.lineWidth = 1;
    octx.moveTo(cx + Math.cos(a) * R * 0.16, cy + Math.sin(a) * R * 0.16);
    octx.lineTo(cx + Math.cos(a) * R * 0.28, cy + Math.sin(a) * R * 0.28);
    octx.stroke();
  }
}

// ==================== 画声频 ====================
const BARS = 52;
const barLevels = new Float32Array(BARS);

function drawWave(now) {
  const { w, h } = fitCanvas(wave, wctx);
  if (!w || !h) return;
  const col = PAL[state] || PAL.idle;
  const data = spectrum();

  wctx.clearRect(0, 0, w, h);

  const slot = w / BARS;
  const bw = Math.max(2, slot * 0.55);

  for (let i = 0; i < BARS; i++) {
    let target;
    if (data) {
      // 对数取样：低频给更多格子，人声集中在这里
      const pos = Math.pow(i / BARS, 1.5);
      const bin = Math.min(data.length - 1, Math.floor(pos * data.length * 0.62));
      target = (data[bin] / 255) * (1 - pos * 0.35);
      if (state === 'idle') target = 0;
    } else {
      target = 0;
    }
    // idle 时给一点极轻的呼吸，表示"在线"
    const idleBreath = 0.05 + 0.035 * Math.sin(now * 1.3 + i * 0.35);
    if (state === 'idle') target = idleBreath;

    barLevels[i] += (target - barLevels[i]) * 0.35;

    const bh = Math.max(2, barLevels[i] * (h - 6));
    const x = i * slot + (slot - bw) / 2;
    const y = h - bh - 2;
    wctx.fillStyle = hsla(col.h, col.s + 6, col.l + 8, 0.28 + barLevels[i] * 0.7);
    if (wctx.roundRect) {
      wctx.beginPath();
      wctx.roundRect(x, y, bw, bh, bw / 2);
      wctx.fill();
    } else {
      wctx.fillRect(x, y, bw, bh);
    }
  }

  // 中线
  wctx.strokeStyle = hsla(col.h, col.s, col.l, 0.12);
  wctx.lineWidth = 1;
  wctx.beginPath();
  wctx.moveTo(0, h - 1.5);
  wctx.lineTo(w, h - 1.5);
  wctx.stroke();
}

// ==================== 主循环 ====================
function drawDiag(now) {
  if (!diagEl.classList.contains('on') || now - diagAt < 0.4) return;
  diagAt = now;
  const t = micTrack;
  let s = {};
  try { if (t && t.getSettings) s = t.getSettings() || {}; } catch (e) {}
  const lines = [
    'ctx       : ' + (audioCtx ? audioCtx.state : '未创建')
      + '   analyser: ' + (analyser ? 'yes' : 'no'),
    'track     : ' + (t ? (t.label || '(无标签)') : '(当前未录音)'),
    'track 状态: readyState=' + (t ? t.readyState : '-')
      + '  muted=' + (t ? t.muted : '-')
      + '  enabled=' + (t ? t.enabled : '-'),
    '采样       : ' + (s.sampleRate || '-') + ' Hz  '
      + (s.channelCount || '-') + ' ch',
    '信号       : rms=' + level().toFixed(4)
      + '  (' + (analyser ? '已接分析器' : '未接 —— 波形不可能动') + ')',
    '状态       : ' + state,
    '可用输入   : ' + devSel.options.length + ' 个',
  ];
  diagEl.textContent = lines.join('\n');
}

function loop() {
  const now = performance.now() / 1000;
  const amp = state === 'idle' ? 0 : level();

  // 录音时：电平条 + 静默告警（"说了话但采不到"最容易误判）
  if (state === 'recording') {
    meterFill.style.width = Math.min(100, amp * 150) + '%';
    if (amp > 0.02) {
      silentSince = 0;
      if (silenceWarned) silenceWarned = false;
    } else if (!silentSince) {
      silentSince = now;
    } else if (!silenceWarned && now - silentSince > 2.5) {
      silenceWarned = true;
      const susp = audioCtx && audioCtx.state !== 'running';
      setHint(susp
        ? '⚠️ 音频上下文被浏览器挂起（' + audioCtx.state + '）—— 点一下页面任意处再试'
        : '⚠️ 没收到声音（电平 0）—— 检查麦克风静音键，或换一个输入设备（当前：' + devLabel + '）');
    }
  }

  // ---- 电话模式：自动断句 + 打断 ----
  if (phoneOn) {
    if (state === 'recording') {
      const a = analyserLevel(micAnalyser);
      if (a > VAD_SPEECH) {
        if (!speechFrom) speechFrom = now;
        hasSpeech = true;
        silenceFrom = 0;
      } else if (hasSpeech) {
        if (!silenceFrom) silenceFrom = now;
        else if (now - silenceFrom > VAD_SILENCE) {
          // 只说了"咳"一声这种极短的，判为噪声，继续听，不发送
          if (now - speechFrom >= MIN_SPEECH) endPhoneTurn();
          else { hasSpeech = false; speechFrom = 0; silenceFrom = 0; }
        }
      }
      if (hasSpeech && now - phoneTurnAt > VAD_MAX_TURN) endPhoneTurn();
      // 一直没人说话：定期重开一段，避免录出一个巨大的空文件
      if (!hasSpeech && now - phoneTurnAt > 40) endPhoneTurn();
    } else if (state === 'speaking') {
      // 她说话时：你开口就可以打断（单向，只允许你打断她）
      const m = analyserLevel(micAnalyser);
      if (m > BARGE_LEVEL) {
        if (!bargeFrom) bargeFrom = now;
        else if (now - bargeFrom > BARGE_HOLD) { interruptHer(); }
      } else {
        bargeFrom = 0;
      }
    }
  }

  drawOrb(now, amp);
  drawWave(now);
  drawDiag(now);
  requestAnimationFrame(loop);
}

// ==================== 界面状态 ====================
// 说明：真流式之后 thinking / speaking 也允许点 —— 那一下不是"禁止说话"，
// 而是"打断她"。对讲机感就是从这里消失的。
const LABEL = { idle: '说话', recording: '发送', thinking: '打断', speaking: '打断' };
let micBlocked = false;          // 组件没就绪（whisper 没加载 / 模型服务连不上）

function setState(s) {
  state = s;
  mic.className = (s === 'recording') ? 'rec'
                : (s === 'thinking') ? 'think'
                : (s === 'speaking') ? 'speak' : '';
  mic.textContent = phoneOn ? '通话中' : LABEL[s];
  // recording 必须可点（那一下 = 发送）；thinking/speaking 也可点（那一下 = 打断）
  mic.disabled = !hasMic || phoneOn || micBlocked;
  orb.style.cursor = mic.disabled ? 'default' : 'pointer';

  if (s === 'recording') {
    recStart = Date.now();
    silentSince = 0;
    silenceWarned = false;
    clearInterval(recTick);
    recTick = setInterval(() => {
      const t = Math.floor((Date.now() - recStart) / 1000);
      const pct = Math.round(Math.min(1, level()) * 100);
      recTimer.textContent = '● 录音 ' + String(Math.floor(t / 60)) + ':' + String(t % 60).padStart(2, '0')
                           + (analyser ? '　电平 ' + pct + '%' : '　(无波形)');
    }, 200);
    recTimer.classList.add('on');
    meterEl.classList.add('on');
    devInfo.classList.add('on');
  } else {
    clearInterval(recTick);
    recTimer.classList.remove('on');
    recTimer.textContent = '';
    meterEl.classList.remove('on');
    meterFill.style.width = '0%';
    if (s !== 'idle') { /* 保留设备名，方便排查 */ } else { devInfo.classList.remove('on'); }
  }
}

function addRow(who, text, cls, live, before) {
  const row = document.createElement('div');
  row.className = 'row ' + cls;
  if (who) {
    const w = document.createElement('div');
    w.className = 'who'; w.textContent = who;
    row.appendChild(w);
  }
  const b = document.createElement('div');
  b.className = 'bub' + (live ? ' live' : '');    // live = 正在流式输出（尾巴有光标）
  b.textContent = text;
  row.appendChild(b);
  // before：插到某一行前面。assistant 气泡是先建好的（流式要用），
  // 用户那行后到，必须插到它前面，否则对话顺序会变成「答在上、问在下」。
  if (before && before.parentNode === logEl) logEl.insertBefore(row, before);
  else logEl.appendChild(row);
  logEl.scrollTop = logEl.scrollHeight;
  // 记录隐藏时，非系统消息计入未读角标
  if (!logOpen && cls !== 'sys') {
    unread += 1;
    renderUnread();
  }
  return b;
}
function setHint(t) { hint.textContent = t; }

// ==================== 健康检查 ====================
async function checkHealth() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    const line = (label, o) =>
      '<span class="' + (o.ok ? 'good' : 'bad') + '">' + label + ' ' + (o.ok ? 'OK' : '×') + '</span>';
    health.innerHTML = [line('STT', d.stt), line('LLM', d.llm), line('TTS', d.tts)].join('　')
      + '<br><span style="font-size:11px">' + d.llm.base + '</span>';

    const allOk = d.stt.ok && d.llm.ok;
    if (allOk) {
      micBlocked = false;
      if (state === 'idle') {
        setState('idle');
        setHint(hasMic ? '点球说话（说完再点一下发送）；她在说时点球可以打断'
                       : '麦克风不可用（' + micWhy() + '）—— 打字也能聊');
      }
    } else if (d.stt.loading) {
      micBlocked = true; mic.disabled = true; mic.textContent = '加载中';
      if (state === 'idle') setHint('语音识别模型正在载入（首次约 10~20 秒）…… 也可以先打字');
    } else {
      micBlocked = true; mic.disabled = true; mic.textContent = '不可用';
      if (state === 'idle') {
        setHint('组件没就绪：'
          + (!d.stt.ok ? '语音识别未加载（' + (d.stt.msg || '') + '）' : '')
          + (!d.llm.ok ? '模型服务连不上' : '') + ' —— 打字仍然可用');
      }
    }
  } catch (e) {
    health.innerHTML = '<span class="bad">后端连不上</span>';
    micBlocked = true; mic.disabled = true; mic.textContent = '离线';
    setHint('后端没起来？确认 run_app.sh 正在运行');
  }
}

// ==================== 输入设备 ====================
const MIC_KEY = 'ev.micId';           // 记住上次选的麦克风，下次打开不用再挑

async function fillDevices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  try {
    const list = await navigator.mediaDevices.enumerateDevices();
    const ins = list.filter(d => d.kind === 'audioinput');
    if (!ins.length) return;
    const cur = inputDeviceId || devSel.value;
    devSel.innerHTML = '';
    ins.forEach((d, i) => {
      const o = document.createElement('option');
      o.value = d.deviceId;
      // 没授权时 label 是空的，只能靠序号区分；点「刷新」授权后就显示真名
      o.textContent = (i + 1) + '. ' + (d.label || '麦克风（未授权，点「刷新」显示名称）');
      devSel.appendChild(o);
    });
    if (cur && ins.some(d => d.deviceId === cur)) devSel.value = cur;
    else if (!cur) inputDeviceId = devSel.value;
    // 注意：未授权时 Chrome 会隐藏 deviceId，这时 cur 匹配不上 ——
    // 但**不要清掉记忆的 ID**，否则每次打开都要重选
    devRow.classList.add('on');
  } catch (e) { /* 没权限时拿不到 label，忽略 */ }
}

function rememberDevice() {
  inputDeviceId = devSel.value;
  try { localStorage.setItem(MIC_KEY, inputDeviceId); } catch (e) {}
}

devSel.onchange = () => {
  rememberDevice();
  const name = devSel.options[devSel.selectedIndex].textContent;
  setHint('已切换到「' + name + '」—— 下次录音生效（已记住）');
};

// 授权一次，把设备真实名称刷出来（Chrome 只有授权后才给 label）
devRefresh.onclick = async () => {
  try {
    const s = await navigator.mediaDevices.getUserMedia({ audio: true });
    s.getTracks().forEach(t => t.stop());
  } catch (e) { setHint('授权失败：' + e.message); return; }
  await fillDevices();
  const sel = devSel.options[devSel.selectedIndex];
  setHint('设备列表已更新，当前：' + (sel ? sel.textContent : ''));
};

// 统一的麦克风约束 + 打开逻辑（记住的设备失效时自动退回默认）
function micConstraints() {
  return Object.assign(
    { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    inputDeviceId ? { deviceId: { exact: inputDeviceId } } : {}
  );
}
async function openMic() {
  try {
    return await navigator.mediaDevices.getUserMedia({ audio: micConstraints() });
  } catch (e) {
    if (e.name === 'OverconstrainedError' && inputDeviceId) {
      inputDeviceId = '';
      try { localStorage.removeItem(MIC_KEY); } catch (err) {}
      return await navigator.mediaDevices.getUserMedia({ audio: micConstraints() });
    }
    throw e;
  }
}
// 把"实际用到的设备"记下来，下次自动用同一个
function adoptDevice(track) {
  const s = (track && track.getSettings) ? track.getSettings() : {};
  if (!s.deviceId) return;
  inputDeviceId = s.deviceId;
  try { localStorage.setItem(MIC_KEY, s.deviceId); } catch (e) {}
  devSel.value = s.deviceId;
}

// ==================== 录音 ====================
async function startRec() {
  if (!hasMic) { setHint('麦克风不可用：' + micWhy()); return; }

  // ⚠️ 必须在"用户点按钮"这个手势里同步创建 AudioContext。
  // 如果等 getUserMedia 的 await 之后再建，浏览器认为手势已过期，
  // ctx 会停在 suspended，AnalyserNode 读出来全是 0 —— 表现就是"录音中但毫无反应"。
  const ctx = ensureCtx();
  if (ctx && ctx.state === 'suspended') { try { await ctx.resume(); } catch (e) {} }

  try {
    const stream = await openMic();
    const graphOk = attachMic(stream);
    const track = stream.getAudioTracks()[0];
    micTrack = track;
    adoptDevice(track);
    devLabel = track ? (track.label || '未知设备') : '无音频轨';
    devInfo.textContent = '输入设备：' + devLabel;
    fillDevices();

    const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus' : '';
    recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    chunks = [];
    recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
    recorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      detachMic();
      sendAudio(new Blob(chunks, { type: recorder.mimeType || 'audio/webm' }));
    };
    recorder.start();
    setState('recording');
    setHint(graphOk
      ? '正在录音…… 说完点「发送」（球和声频在动就说明收到了）'
      : '正在录音，但音频分析不可用（' + audioGraphErr + '）—— 先发出去看识别结果');
  } catch (e) {
    setState('idle');
    setHint('拿不到麦克风：' + e.message + '　（'
      + (micWhy() || '检查系统麦克风权限：Windows 设置 → 隐私 → 麦克风') + '）');
  }
}
function stopRec() {
  if (recorder && recorder.state === 'recording') recorder.stop();
}

// ==================== 电话模式（连续对话） ====================
// 原理：麦克风一直开着 -> 录音分段 -> 检测到"说过话 + 停顿 0.75s"就自动发送
//      -> 她说完自动重新开始收音；说话期间检测到你的声音就打断她（barge-in）

// 手机和 PC 的声学环境完全不同：扬声器和麦克风贴在一起（回声直接进麦克风）、
// 系统 AGC 还会把底噪一起抬起来。同一套阈值在手机上只会表现为两种故障：
//   「她还没说完就被自己的回声打断」和「你还没说完就被切成发送」。
// 所以按设备走两套预设 —— 这不是调优，是同一个算法在两个声学环境下必须的标定。
const IS_MOBILE = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent)
               || !!(window.matchMedia && matchMedia('(pointer:coarse)').matches);
// 想手动覆盖：地址后加 ?gain=mobile / ?gain=pc（排查时很有用）
const _gain = (new URLSearchParams(location.search).get('gain') || '').toLowerCase();
const MOBILE_PRESET = _gain ? (_gain === 'mobile' || _gain === '手机') : IS_MOBILE;

const VAD_SPEECH = MOBILE_PRESET ? 0.10 : 0.055;  // 判定"在说话"的音量阈值
const VAD_SILENCE = MOBILE_PRESET ? 1.2 : 1.4;    // 停顿多久才认为你说完
const MIN_SPEECH = MOBILE_PRESET ? 0.40 : 0.35;   // 至少说这么久才算一句（挡咳嗽/碰麦）
const VAD_MAX_TURN = 20;                          // 单轮最长(秒)
const BARGE_LEVEL = MOBILE_PRESET ? 0.24 : 0.12;  // 打断阈值（回声大时必须抬高）
const BARGE_HOLD = MOBILE_PRESET ? 0.40 : 0.25;   // 持续多久算真的打断
const POST_PLAY_GUARD = MOBILE_PRESET ? 520 : 260; // 她说完到重新收音的间隔(ms)，防尾音

async function startPhone() {
  if (!hasMic) { setHint('麦克风不可用：' + micWhy()); return; }
  ensureCtx();
  try {
    phoneStream = await openMic();
  } catch (e) {
    setHint('打不开麦克风：' + e.message);
    return;
  }
  attachMic(phoneStream);
  const track = phoneStream.getAudioTracks()[0];
  micTrack = track;
  adoptDevice(track);
  devLabel = track ? (track.label || '未知设备') : '无音频轨';
  devInfo.textContent = '输入设备：' + devLabel
    + '　·　' + (MOBILE_PRESET ? '手机预设（回声阈值已抬高）' : 'PC 预设');
  devInfo.classList.add('on');
  fillDevices();

  phoneOn = true;
  phoneBtn.classList.add('on');
  phoneBtn.textContent = '挂断';
  mic.disabled = true;
  beginPhoneTurn();
}

function beginPhoneTurn() {
  if (!phoneOn || !phoneStream) return;
  hasSpeech = false;
  speechFrom = 0;
  silenceFrom = 0;
  bargeFrom = 0;
  phoneChunks = [];

  const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    ? 'audio/webm;codecs=opus' : '';
  try {
    phoneRec = new MediaRecorder(phoneStream, mime ? { mimeType: mime } : undefined);
  } catch (e) {
    setHint('录音器启动失败：' + e.message);
    return;
  }
  phoneRec.ondataavailable = e => { if (e.data.size) phoneChunks.push(e.data); };
  phoneRec.onstop = () => {
    const blob = new Blob(phoneChunks, { type: phoneRec.mimeType || 'audio/webm' });
    if (!phoneOn) { setState('idle'); return; }
    if (blob.size < 3000) { beginPhoneTurn(); return; }   // 太短，当噪声，重来
    sendAudio(blob);
  };
  phoneRec.start();
  phoneTurnAt = performance.now() / 1000;
  setState('recording');
  setHint('在听…… 慢慢说（她不会抢话）；她说话时你可随时插话');
}

function endPhoneTurn() {
  if (phoneRec && phoneRec.state === 'recording') phoneRec.stop();
}

// 重新开始收音（带一点延迟，避免她播放的尾音/回声被当成你说的话）
function resumeListening(delay) {
  if (!phoneOn) return;
  setTimeout(() => { if (phoneOn) beginPhoneTurn(); },
             delay == null ? POST_PLAY_GUARD : delay);
}

// 打断她：停播 + 掐掉还没合完的 TTS + 立刻回到收音（流式之后这是常态操作）
function interruptHer() {
  abortTurn();
  bargeFrom = 0;
  if (phoneOn) beginPhoneTurn();
  else setState('idle');
}

function stopPhone() {
  phoneOn = false;
  phoneBtn.classList.remove('on');
  phoneBtn.textContent = '电话';
  abortTurn();
  try { if (phoneRec && phoneRec.state === 'recording') phoneRec.stop(); } catch (e) {}
  if (phoneStream) {
    phoneStream.getTracks().forEach(t => t.stop());
    phoneStream = null;
  }
  detach();
  setState('idle');
  setHint('已挂断');
}

phoneBtn.onclick = () => { phoneOn ? stopPhone() : startPhone(); };

// ==================== 一轮对话（SSE 流式） ====================
// 服务端事件: meta(识别结果) / delta(增量文字) / audio(一段 mp3) / done / error
// 前端只做三件事：把文字拼进气泡、把音频排进播放队列、根据状态更新 UI。

let streamToken = 0;        // 每轮 +1：旧请求的回调看到这个变了就自我失效
let activeCtrl = null;      // 当前这轮的 AbortController（打断就是 abort 它）
let turn = null;            // { token, reply, done, timing }

function isToken(tok) { return !!turn && turn.token === tok; }

function newTurn() {
  streamToken += 1;
  turn = { token: streamToken, reply: '', done: false, timing: null, finished: false };
  return turn;
}

// SSE 读取器：fetch + ReadableStream 手搓，因为 EventSource 不能 POST 音频
async function streamTurn(url, body, hintText) {
  const t = newTurn();
  if (activeCtrl) { try { activeCtrl.abort(); } catch (e) {} }
  const ctrl = new AbortController();
  activeCtrl = ctrl;
  stopPlayback();

  setState('thinking');
  setHint(hintText || '思考中……');
  const bubble = addRow('E.V.', '', 'ev', true);
  t.bubble = bubble;          // 记在本轮上：被打断时要能把它收拾干净

  try {
    const isForm = (typeof FormData !== 'undefined') && (body instanceof FormData);
    const res = await fetch(url, {
      method: 'POST', body: body, signal: ctrl.signal,
      headers: isForm ? undefined : { 'Content-Type': 'application/json' },
    });
    if (!res.ok || !res.body) throw new Error('HTTP ' + res.status);

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        let ev;
        try { ev = JSON.parse(payload); } catch (e) { continue; }
        if (!isToken(t.token)) return;
        onEvent(t, ev, bubble);
      }
    }
    // 连接正常结束但没收到 done（后端异常断流）
    if (isToken(t.token) && !t.done && !t.finished) {
      t.done = true;
      if (!player.srcs.length) finishTurn(t);
    }
  } catch (e) {
    if (!isToken(t.token)) return;                    // 被打断引起的 abort，不算错误
    if (bubble.parentNode) bubble.parentNode.remove();
    setState('idle');
    setHint('请求失败：' + e.message + '（serve_hf.py 起了吗？）');
    if (phoneOn) resumeListening(800);
  }
}

function onEvent(t, ev, bubble) {
  if (ev.t === 'meta') {
    // 插到 E.V. 那行前面：问句必须在答句上方（自上而下）
    if (ev.user_text) addRow('你', ev.user_text, 'me', false, bubble.parentElement);
    setHint('在想……');
  } else if (ev.t === 'delta') {
    t.reply += ev.d;
    bubble.textContent = t.reply;
    if (logOpen) logEl.scrollTop = logEl.scrollHeight;
  } else if (ev.t === 'audio') {
    enqueueAudio(ev.b64, t.token);                    // 不 await：边播边收下一段
  } else if (ev.t === 'done') {
    if (ev.reply) { t.reply = ev.reply; bubble.textContent = t.reply; }
    bubble.classList.remove('live');
    t.done = true;
    t.timing = ev.timing || {};
    if (ev.tts_error) t.ttsErr = ev.tts_error;
    if (!player.srcs.length && !pendingAudio) finishTurn(t);  // 极端情况：TTS 全失败
  } else if (ev.t === 'error') {
    bubble.classList.remove('live');
    setState('idle');
    setHint(ev.msg || '出错了');
    t.finished = true;
    if (phoneOn) resumeListening(600);
  }
}

// 最后一段音频播完（或没音频可播）时收尾
function onSpeechDrained() {
  if (!turn || !turn.done) return;                    // token 还在来，先别收尾
  if (pendingAudio > 0) return;                       // 还有段没解完码
  finishTurn(turn);
}

function finishTurn(t) {
  if (!isToken(t.token) || t.finished) return;
  t.finished = true;
  const tm = t.timing || {};
  const cost = tm.first_audio
    ? '（首声 ' + tm.first_audio + 's / 总 ' + tm.total + 's）' : '';
  if (phoneOn) { resumeListening(); return; }         // 电话模式：接着说
  setState('idle');
  setHint(t.ttsErr
    ? '语音合成失败，这条只有文字'
    : '可以继续说了　' + cost + '　嫌点着麻烦就开「电话」');
}

function sendAudio(blob) {
  const fd = new FormData();
  fd.append('audio', blob, 'voice.webm');
  streamTurn('/api/stream/turn', fd, '识别中……');
}

function sendText(text) {
  text = (text || '').trim();
  if (!text) return;
  if (phoneOn) { setHint('电话模式下直接说话就行（想打字先点「挂断」）'); return; }
  ensureCtx();                                        // 在按键手势里唤醒音频上下文
  if (state !== 'idle') abortTurn();                  // 正在说/正在想：直接打断再问
  streamTurn('/api/stream/text', JSON.stringify({ text: text }), '思考中……');
}

// 打断：停播 + abort 请求（后端 generator 被取消 -> 模型停止生成）+ 作废本轮令牌
function abortTurn() {
  if (activeCtrl) { try { activeCtrl.abort(); } catch (e) {} activeCtrl = null; }
  stopPlayback();
  // 收尾被打断那一轮的行：一个字都没说出来就删掉，说了半句就留着标「被打断」
  const b = turn && turn.bubble;
  if (b && b.parentElement) {
    if (!b.textContent.trim()) b.parentElement.remove();
    else { b.classList.remove('live'); b.classList.add('cut'); }
  }
  turn = null;
  streamToken += 1;
}

// ==================== 记录面板（默认隐藏） ====================
function renderUnread() {
  if (unread > 0) {
    unreadEl.textContent = unread > 9 ? '9+' : String(unread);
    unreadEl.classList.add('on');
  } else {
    unreadEl.classList.remove('on');
  }
}

function setLogOpen(open) {
  logOpen = open;
  logEl.classList.toggle('show', open);
  logEl.setAttribute('aria-hidden', String(!open));
  logToggle.setAttribute('aria-pressed', String(open));
  // 宽屏时记录是右侧常驻面板，靠 body.nolog 收起整列
  document.body.classList.toggle('nolog', !open);
  if (open) {
    unread = 0;
    renderUnread();
    logEl.scrollTop = logEl.scrollHeight;
  }
}

logToggle.onclick = () => setLogOpen(!logOpen);

// ==================== 事件 ====================
function onTap() {
  if (phoneOn) return;
  if (state === 'idle') startRec();
  else if (state === 'recording') stopRec();
  else { abortTurn(); startRec(); }        // 她在说 / 在想：这一下 = 打断 + 我要说
}
mic.onclick = onTap;
orb.onclick = onTap;

document.getElementById('reset').onclick = async () => {
  await fetch('/api/reset', { method: 'POST' });
  logEl.innerHTML = '';
  addRow('', '已清空上下文', 'sys');
};

typeIn.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.isComposing) { sendText(typeIn.value); typeIn.value = ''; }
});

document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && logOpen) setLogOpen(false);
});

// 球体本身可聚焦，键盘也能开始/结束说话
orb.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    onTap();
  }
});

// 任何一次点击都尝试唤醒被挂起的 AudioContext（浏览器自动播放策略）
document.addEventListener('pointerdown', () => {
  if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
}, true);

buildStamp.onclick = () => diagEl.classList.toggle('on');

if (!hasMic) {
  mic.textContent = '无麦克风';
  mic.disabled = true;
  addRow('', '麦克风不可用：' + micWhy() + '　已切换为键盘输入模式（回复照常朗读）。', 'sys');
}

addRow('', '点中间的球开始说话。她话不多，别指望长篇大论。'
  + (MOBILE_PRESET ? '　[手机预设：回声阈值已抬高]' : ''), 'sys');
setState('idle');
micBlocked = true;         // 等健康检查确认组件就绪后再放开
mic.disabled = true;

// 恢复上次选的麦克风（不用每次先录一遍才改设备）
try { inputDeviceId = localStorage.getItem(MIC_KEY) || ''; } catch (e) {}
fillDevices();

requestAnimationFrame(loop);
checkHealth();
setInterval(() => { if (state === 'idle') checkHealth(); }, 8000);
// 宽屏：对话记录常显（JARVIS 那种右侧终端）；窄屏：默认收起，点「记录」才弹出
setLogOpen(window.innerWidth > 900);
window.addEventListener('resize', () => { /* 尺寸在 draw 里自适应 */ paintTelemetry(); });
