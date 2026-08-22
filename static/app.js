const video = document.querySelector('#preview');
const startButton = document.querySelector('#start');
const count = document.querySelector('#count');
const ring = document.querySelector('#ring');
const motionWarning = document.querySelector('#motion');
const toast = document.querySelector('#toast');
const headline = document.querySelector('#headline');
const detail = document.querySelector('#detail');
const sessionId = new URLSearchParams(location.search).get('session') || 'demo';
document.querySelector('#session').textContent = `SESSION ${sessionId.toUpperCase()}`;

const config = await fetch('/api/config', { cache: 'no-store' }).then((response) => {
  if (!response.ok) throw new Error('Could not load capture configuration');
  return response.json();
});
const clientId = localStorage.roomscanClientId || crypto.randomUUID().replaceAll('-', '').slice(0, 12);
localStorage.roomscanClientId = clientId;
const canvas = document.createElement('canvas');
const context = canvas.getContext('2d', { willReadFrequently: true });
let stream;
let capturing = false;
let candidates = [];
let rotationTravel = 0;
let translationSignal = 0;
let lastMotionTime = 0;
let captureStarted = 0;

function showToast(html) {
  toast.innerHTML = html;
  toast.hidden = false;
}

async function requestMotionPermission() {
  const permissionTypes = [globalThis.DeviceOrientationEvent, globalThis.DeviceMotionEvent];
  for (const permissionType of permissionTypes) {
    if (typeof permissionType?.requestPermission === 'function') {
      const state = await permissionType.requestPermission();
      if (state !== 'granted') throw new Error('Motion permission is required to check camera movement.');
    }
  }
}

function handleMotion(event) {
  if (!capturing) return;
  const now = event.timeStamp || performance.now();
  const dt = lastMotionTime ? Math.min((now - lastMotionTime) / 1000, 0.2) : 0;
  lastMotionTime = now;
  const rotation = event.rotationRate || {};
  const rotationMagnitude = Math.hypot(rotation.alpha || 0, rotation.beta || 0, rotation.gamma || 0);
  rotationTravel += rotationMagnitude * dt;
  const acceleration = event.acceleration || {};
  const accelerationMagnitude = Math.hypot(acceleration.x || 0, acceleration.y || 0, acceleration.z || 0);
  translationSignal += accelerationMagnitude * dt;
  const oldEnough = performance.now() - captureStarted > 3000;
  const pureRotation = rotationTravel > config.motion_rotation_threshold && translationSignal < config.motion_translation_threshold;
  motionWarning.classList.toggle('show', oldEnough && pureRotation);
}

function blurScore(imageData) {
  const { data, width, height } = imageData;
  let sum = 0;
  let sumSquares = 0;
  let samples = 0;
  const gray = (x, y) => {
    const index = (y * width + x) * 4;
    return data[index] * 0.299 + data[index + 1] * 0.587 + data[index + 2] * 0.114;
  };
  for (let y = 2; y < height - 2; y += 4) {
    for (let x = 2; x < width - 2; x += 4) {
      const laplacian = gray(x - 1, y) + gray(x + 1, y) + gray(x, y - 1) + gray(x, y + 1) - 4 * gray(x, y);
      sum += laplacian;
      sumSquares += laplacian * laplacian;
      samples += 1;
    }
  }
  const mean = sum / Math.max(samples, 1);
  return sumSquares / Math.max(samples, 1) - mean * mean;
}

function canvasBlob() {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error('JPEG encoding failed')), 'image/jpeg', config.jpeg_quality);
  });
}

async function sampleFrame() {
  const width = config.capture_width;
  const height = Math.round(width * video.videoHeight / video.videoWidth);
  canvas.width = width;
  canvas.height = height;
  context.drawImage(video, 0, 0, width, height);
  const score = blurScore(context.getImageData(0, 0, width, height));
  const blob = await canvasBlob();
  candidates.push({ score, blob });
  const bufferLimit = Math.max(config.frames_per_client * 2, Math.ceil(config.capture_seconds * config.capture_fps));
  if (candidates.length > bufferLimit) candidates.shift();
}

function waitForSocket(socket, expectedType) {
  return new Promise((resolve, reject) => {
    const handleMessage = (event) => {
      const value = JSON.parse(event.data);
      if (value.type === 'error') { cleanup(); reject(new Error(value.message)); }
      else if (value.type === expectedType) { cleanup(); resolve(value); }
    };
    const handleClose = () => { cleanup(); reject(new Error('Upload connection closed early')); };
    const cleanup = () => {
      socket.removeEventListener('message', handleMessage);
      socket.removeEventListener('close', handleClose);
    };
    socket.addEventListener('message', handleMessage);
    socket.addEventListener('close', handleClose);
  });
}

async function uploadFrames(frames) {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/ws/upload`);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', () => reject(new Error('Could not connect to upload server')), { once: true });
  });
  socket.send(JSON.stringify({ type: 'start', session_id: sessionId, client_id: clientId, frame_count: frames.length }));
  await waitForSocket(socket, 'ready');
  for (let index = 0; index < frames.length; index += 1) {
    detail.textContent = `Uploading ${index + 1} of ${frames.length}…`;
    socket.send(await frames[index].blob.arrayBuffer());
    await waitForSocket(socket, 'ack');
  }
  socket.send(JSON.stringify({ type: 'complete' }));
  return waitForSocket(socket, 'complete');
}

async function beginCapture() {
  startButton.disabled = true;
  try {
    await requestMotionPermission();
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
    });
    video.srcObject = stream;
    await video.play();
    if (!video.videoWidth) throw new Error('Camera did not become ready');
  } catch (error) {
    startButton.disabled = false;
    showToast(`${error.message}<br><br>Camera and motion access require HTTPS. Reload to try again.`);
    return;
  }
  candidates = [];
  rotationTravel = 0;
  translationSignal = 0;
  lastMotionTime = 0;
  captureStarted = performance.now();
  capturing = true;
  addEventListener('devicemotion', handleMotion);
  startButton.hidden = true;
  count.hidden = false;
  headline.textContent = 'Keep moving sideways';
  detail.textContent = 'Aim steadily at walls, corners, and furniture.';
  const durationMs = config.capture_seconds * 1000;
  const sampleEveryMs = 1000 / config.capture_fps;
  let nextSample = 0;

  await new Promise((resolve) => {
    const tick = async (now) => {
      const elapsed = now - captureStarted;
      const progress = Math.min(elapsed / durationMs, 1);
      ring.style.strokeDashoffset = String(527.8 * (1 - progress));
      count.textContent = String(Math.max(0, Math.ceil(config.capture_seconds - elapsed / 1000)));
      if (elapsed >= nextSample && elapsed < durationMs) {
        nextSample += sampleEveryMs;
        await sampleFrame();
      }
      if (elapsed < durationMs) requestAnimationFrame(tick); else resolve();
    };
    requestAnimationFrame(tick);
  });
  capturing = false;
  removeEventListener('devicemotion', handleMotion);
  motionWarning.classList.remove('show');
  count.textContent = '✓';
  headline.textContent = 'Selecting sharp frames';
  const sharp = candidates
    .sort((a, b) => b.score - a.score)
    .slice(0, config.frames_per_client);
  const passing = sharp.filter((frame) => frame.score >= config.blur_threshold).length;
  detail.textContent = `${passing}/${sharp.length} frames passed the sharpness target. Uploading…`;
  try {
    const result = await uploadFrames(sharp);
    showToast(`<strong>Upload complete</strong><br>${result.frames} sharp frames joined session <b>${sessionId}</b>.<br><br>You can put your phone away.`);
  } catch (error) {
    showToast(`<strong>Upload failed</strong><br>${error.message}<br><br>Your captured frames remain in this tab; reload to retry the scan.`);
  } finally {
    stream?.getTracks().forEach((track) => track.stop());
  }
}

startButton.addEventListener('click', beginCapture);
