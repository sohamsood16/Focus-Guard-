import { FaceLandmarker, ObjectDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const OBJECT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";

const video = document.querySelector("#webcam");
const canvas = document.querySelector("#overlay");
const ctx = canvas.getContext("2d");
const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const pauseBtn = document.querySelector("#pauseBtn");
const alarmTestBtn = document.querySelector("#alarmTestBtn");
const taskInput = document.querySelector("#taskInput");
const taskDisplay = document.querySelector("#taskDisplay");
const dashboard = document.querySelector("#dashboard");
const setupCard = document.querySelector("#setupCard");
const systemStatus = document.querySelector("#systemStatus");
const alarmOverlay = document.querySelector("#alarmOverlay");
const alarmReason = document.querySelector("#alarmReason");
const scoreEl = document.querySelector("#score");
const scoreBar = document.querySelector("#scoreBar");
const stateText = document.querySelector("#stateText");
const faceMetric = document.querySelector("#faceMetric");
const headMetric = document.querySelector("#headMetric");
const eyeMetric = document.querySelector("#eyeMetric");
const phoneMetric = document.querySelector("#phoneMetric");
const streakMetric = document.querySelector("#streakMetric");
const alertsMetric = document.querySelector("#alertsMetric");
const eventLog = document.querySelector("#eventLog");

let landmarker;
let objectDetector;
let stream;
let running = false;
let paused = false;
let lastVideoTime = -1;
let animationId = 0;
let distractionSince = null;
let lastAlertAt = 0;
let alerts = 0;
let lastReason = "";
let alertedThisDistraction = false;
let audioCtx;
let alarmTimer = null;
let calibration = { yaw0: 0, pitch0: 0, samples: [] };
let lastFrameAt = performance.now();

function logEvent(message) {
  const now = new Date().toLocaleTimeString();
  const line = document.createElement("div");
  line.textContent = `[${now}] ${message}`;
  eventLog.prepend(line);
  while (eventLog.children.length > 30) eventLog.lastElementChild.remove();
}

function setStatus(text, active=false) {
  systemStatus.textContent = text;
  systemStatus.style.color = active ? "#79f0b0" : "#98a6c4";
  systemStatus.style.borderColor = active ? "#1d7b57" : "#27314a";
}

function ensureAudio() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") audioCtx.resume();
}

function beep(duration=220, frequency=880) {
  ensureAudio();
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = "square";
  osc.frequency.value = frequency;
  gain.gain.setValueAtTime(0.0001, audioCtx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.25, audioCtx.currentTime + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + duration/1000);
  osc.connect(gain).connect(audioCtx.destination);
  osc.start();
  osc.stop(audioCtx.currentTime + duration/1000 + 0.03);
}

function startAlarm(reason) {
  const now = performance.now();
  // Trigger only once per continuous distraction episode.
  // The overlay then hides automatically; it will not keep reappearing
  // until the user returns to a focused state.
  if (alertedThisDistraction || now - lastAlertAt < 4500) return;
  lastAlertAt = now;
  alertedThisDistraction = true;
  alerts += 1;
  alertsMetric.textContent = alerts;
  lastReason = reason;
  alarmReason.textContent = reason;
  alarmOverlay.hidden = false;
  stateText.textContent = "DISTRACTED — return to your work";
  stateText.style.color = "#ff6c7d";
  logEvent(`ALERT: ${reason}`);
  ensureAudio();
  beep(320, 1100);
  setTimeout(() => beep(320, 780), 370);
  alarmTimer = setTimeout(() => { alarmOverlay.hidden = true; }, 1500);
}

function stopAlarmVisual() {
  alarmOverlay.hidden = true;
  if (alarmTimer) clearTimeout(alarmTimer);
  alarmTimer = null;
}

function calibrationAngle(matrix, axis) {
  // MediaPipe gives a 4x4 transformation matrix as a flat array.
  // We extract a practical yaw/pitch estimate from its rotation submatrix.
  const m = matrix;
  const r00 = m[0], r10 = m[4], r20 = m[8], r21 = m[9], r22 = m[10];
  const yaw = Math.atan2(r20, Math.sqrt(r00*r00 + r10*r10)) * 180/Math.PI;
  const pitch = Math.atan2(-r21, r22) * 180/Math.PI;
  return axis === "yaw" ? yaw : pitch;
}

function blendshapeMap(categories) {
  const map = {};
  for (const c of categories || []) map[c.categoryName] = c.score;
  return map;
}

function estimateEyeDirection(bs) {
  const left = (bs.eyeLookInLeft||0) + (bs.eyeLookOutLeft||0) + (bs.eyeLookUpLeft||0) + (bs.eyeLookDownLeft||0);
  const right = (bs.eyeLookInRight||0) + (bs.eyeLookOutRight||0) + (bs.eyeLookUpRight||0) + (bs.eyeLookDownRight||0);
  const lookLeft = (bs.eyeLookOutRight||0) + (bs.eyeLookInLeft||0);
  const lookRight = (bs.eyeLookOutLeft||0) + (bs.eyeLookInRight||0);
  const lookUp = (bs.eyeLookUpLeft||0) + (bs.eyeLookUpRight||0);
  const lookDown = (bs.eyeLookDownLeft||0) + (bs.eyeLookDownRight||0);
  const total = left + right + 1e-6;
  if (lookUp / total > 0.55) return { label:"UP", away:true, strength: lookUp/total };
  if (lookDown / total > 0.55) return { label:"DOWN", away:true, strength: lookDown/total };
  if (lookLeft / total > 0.52) return { label:"LEFT", away:true, strength: lookLeft/total };
  if (lookRight / total > 0.52) return { label:"RIGHT", away:true, strength: lookRight/total };
  return { label:"CENTER", away:false, strength:0 };
}

function scoreAttention({ face, headAway, eyeAway, eyesClosed }) {
  if (!face) return 0;
  let score = 100;
  if (headAway) score -= 42;
  if (eyeAway) score -= 28;
  if (eyesClosed) score -= 55;
  return Math.max(0, Math.min(100, score));
}

function resizeCanvas() {
  const rect = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);
}
window.addEventListener("resize", resizeCanvas);

function drawOverlay(landmarks, score, status) {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.clearRect(0,0,w,h);
  if (!landmarks) return;
  ctx.lineWidth = 2;
  ctx.strokeStyle = score < 55 ? "#ff6374" : "#62e3a0";
  const pts = [1, 33, 263, 61, 291, 152];
  for (const idx of pts) {
    const p = landmarks[idx];
    if (!p) continue;
    ctx.beginPath();
    ctx.arc(p.x*w, p.y*h, 3, 0, Math.PI*2);
    ctx.stroke();
  }
  ctx.font = "700 13px system-ui";
  ctx.fillStyle = score < 55 ? "#ffb6be" : "#b8ffd8";
  ctx.fillText(status, 14, 24);
}

async function initLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(WASM_URL);
  const options = {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numFaces: 3,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  };
  landmarker = await FaceLandmarker.createFromOptions(filesetResolver, options);
  objectDetector = await ObjectDetector.createFromOptions(filesetResolver, {
    baseOptions: { modelAssetPath: OBJECT_MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    maxResults: 5,
    scoreThreshold: 0.45,
  });
}

async function start() {
  try {
    ensureAudio();
    startBtn.disabled = true;
    startBtn.textContent = "Loading AI…";
    setStatus("LOADING");
    await initLandmarker();
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal:1280 }, height:{ ideal:720 }, facingMode:"user" }, audio:false });
    video.srcObject = stream;
    await video.play();
    dashboard.hidden = false;
    setupCard.hidden = true;
    taskDisplay.textContent = taskInput.value.trim() || "Complete your assigned work";
    resizeCanvas();
    running = true;
    paused = false;
    pauseBtn.textContent = "Pause";
    setStatus("MONITORING", true);
    logEvent("Monitoring started");
    calibration = { yaw0:0, pitch0:0, samples:[] };
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    alert("Could not start webcam/AI. Make sure camera permission is allowed and open this app on localhost (not file://).\n\n" + err.message);
    startBtn.disabled = false;
    startBtn.textContent = "Start monitoring";
    setStatus("OFF");
  }
}

function stop() {
  running = false;
  paused = false;
  cancelAnimationFrame(animationId);
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  video.srcObject = null;
  if (landmarker) { try { landmarker.close(); } catch {} }
  if (objectDetector) { try { objectDetector.close(); } catch {} }
  landmarker = null;
  objectDetector = null;
  alarmOverlay.hidden = true;
  alertedThisDistraction = false;
  setupCard.hidden = false;
  dashboard.hidden = true;
  startBtn.disabled = false;
  startBtn.textContent = "Start monitoring";
  setStatus("OFF");
  logEvent("Monitoring stopped");
}

async function loop(now) {
  if (!running) return;
  animationId = requestAnimationFrame(loop);
  if (paused || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  const result = landmarker.detectForVideo(video, now);
  const objects = objectDetector ? objectDetector.detectForVideo(video, now) : { detections: [] };
  const faces = result.faceLandmarks || [];
  const phoneDetected = (objects.detections || []).some(d => (d.categories || []).some(c => String(c.categoryName || "").toLowerCase().includes("cell phone")));
  phoneMetric.textContent = phoneDetected ? "YES" : "NO";
  if (!faces.length) {
    faceMetric.textContent = "NO";
    headMetric.textContent = "—";
    eyeMetric.textContent = "—";
    const since = distractionSince ?? now;
    if (distractionSince === null) distractionSince = now;
    const sec = (now - since)/1000;
    streakMetric.textContent = sec.toFixed(1)+"s";
    scoreEl.textContent = "0";
    scoreBar.style.width = "0%";
    stateText.textContent = "FACE NOT DETECTED";
    stateText.style.color = "#ff6c7d";
    if (sec > 2.5) startAlarm("Face not detected — return to your workstation");
    drawOverlay(null, 0, "NO FACE");
    return;
  }

  const lm = faces[0];
  const bs = blendshapeMap((result.faceBlendshapes?.[0]?.categories) || []);
  const yaw = result.facialTransformationMatrixes?.[0] ? calibrationAngle(result.facialTransformationMatrixes[0], "yaw") : 0;
  const pitch = result.facialTransformationMatrixes?.[0] ? calibrationAngle(result.facialTransformationMatrixes[0], "pitch") : 0;

  // First ~1 second: learn the user's natural centered posture.
  if (calibration.samples.length < 30) {
    calibration.samples.push({ yaw, pitch });
    calibration.yaw0 = calibration.samples.reduce((s,x)=>s+x.yaw,0)/calibration.samples.length;
    calibration.pitch0 = calibration.samples.reduce((s,x)=>s+x.pitch,0)/calibration.samples.length;
  }

  const yawDelta = yaw - calibration.yaw0;
  const pitchDelta = pitch - calibration.pitch0;
  const headAway = Math.abs(yawDelta) > 22 || Math.abs(pitchDelta) > 18;
  const eye = estimateEyeDirection(bs);
  const blink = ((bs.eyeBlinkLeft||0) + (bs.eyeBlinkRight||0))/2;
  const eyesClosed = blink > 0.72;
  const face = true;
  const score = scoreAttention({ face, headAway, eyeAway: eye.away, eyesClosed });

  faceMetric.textContent = faces.length === 1 ? "YES" : `${faces.length} FACES`;
  headMetric.textContent = headAway ? (yawDelta > 0 ? "RIGHT" : "LEFT") : "CENTER";
  eyeMetric.textContent = eyesClosed ? "CLOSED" : eye.label;
  scoreEl.textContent = Math.round(score);
  scoreBar.style.width = score + "%";
  scoreBar.style.background = score < 55 ? "#ff6374" : score < 75 ? "#f3c969" : "#5be39a";

  const multipleFaces = faces.length > 1;
  const distracted = multipleFaces || phoneDetected || (score < 55 && (headAway || eye.away || eyesClosed));
  if (distracted) {
    if (distractionSince === null) { distractionSince = now; logEvent("Potential distraction detected"); }
    const sec = (now - distractionSince)/1000;
    streakMetric.textContent = sec.toFixed(1)+"s";
    stateText.textContent = sec > 2.5 ? "DISTRACTED — alarm triggered" : "Potential distraction";
    stateText.style.color = sec > 2.5 ? "#ff6c7d" : "#f3c969";
    if (sec > 2.5) {
      let reason = "Look back at your work";
      if (multipleFaces) reason = "More than one person detected";
      else if (phoneDetected) reason = "Phone detected in camera view";
      if (eyesClosed) reason = "Eyes closed for too long";
      else if (headAway) reason = "Head turned away from the screen";
      else if (eye.away) reason = `Eyes looking ${eye.label.toLowerCase()}`;
      startAlarm(reason);
    }
  } else {
    if (distractionSince !== null) logEvent("Focus restored");
    distractionSince = null;
    alertedThisDistraction = false;
    streakMetric.textContent = "0.0s";
    stopAlarmVisual();
    stateText.textContent = "FOCUSED";
    stateText.style.color = "#72edb0";
  }

  drawOverlay(lm, score, distracted ? "FOCUS CHECK" : "FOCUSED");
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
pauseBtn.addEventListener("click", () => {
  paused = !paused;
  pauseBtn.textContent = paused ? "Resume" : "Pause";
  setStatus(paused ? "PAUSED" : "MONITORING", !paused);
  logEvent(paused ? "Monitoring paused" : "Monitoring resumed");
});
alarmTestBtn.addEventListener("click", () => startAlarm("Test alert — FocusGuard alarm is working"));
window.addEventListener("beforeunload", () => stream?.getTracks().forEach(t=>t.stop()));
