const video = document.querySelector('#preview');
const startButton = document.querySelector('#start');
const count = document.querySelector('#count');
const ring = document.querySelector('#ring');
const motionWarning = document.querySelector('#motion');
const toast = document.querySelector('#toast');
const toastMessage = document.querySelector('#toast-message');
const toastAction = document.querySelector('#toast-action');
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
let sampleIndex = 0;

function showToast(html, actionLabel = 'SCAN ANOTHER AREA') {
  toastMessage.innerHTML = html;
  toastAction.textContent = actionLabel;
  toast.hidden = false;
}

function resetForAnotherCapture() {
  toast.hidden = true;
  startButton.hidden = false;
  startButton.disabled = false;
  startButton.textContent = 'START';
  count.hidden = true;
  count.textContent = String(config.capture_seconds);
  ring.style.strokeDashoffset = '527.8';
  headline.textContent = 'Start on the shared landmark';
  detail.textContent = 'All phones: landscape, frame the same distinctive corner, then press START.';
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

function updateCaptureWarning(pureRotation = false) {
  const portraitVideo = video.videoHeight > video.videoWidth;
  if (portraitVideo) {
    motionWarning.innerHTML = 'TURN PHONE SIDEWAYS<br><small>Landscape keeps the floor and ceiling in view.</small>';
    motionWarning.classList.add('show');
  } else if (pureRotation) {
    motionWarning.innerHTML = 'WALK SIDEWAYS<br><small>Rotating in place cannot create depth.</small>';
    motionWarning.classList.add('show');
  } else {
    motionWarning.classList.remove('show');
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
  updateCaptureWarning(oldEnough && pureRotation);
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
  candidates.push({ score, blob, index: sampleIndex });
  sampleIndex += 1;
  const bufferLimit = Math.max(config.frames_per_client * 2, Math.ceil(config.capture_seconds * config.capture_fps));
  if (candidates.length > bufferLimit) candidates.shift();
}

function selectTemporalFrames(frames, limit) {
  if (frames.length <= limit) return frames;
  if (config.frame_selection === 'sharpest') {
    return [...frames].sort((a, b) => b.score - a.score).slice(0, limit)
      .sort((a, b) => a.index - b.index);
  }
  const selected = [];
  for (let window = 0; window < limit; window += 1) {
    const start = Math.floor(window * frames.length / limit);
    const end = Math.floor((window + 1) * frames.length / limit);
    selected.push(frames.slice(start, end).reduce(
      (best, frame) => (frame.score > best.score ? frame : best),
    ));
  }
  return selected;
}

function preferredVideoMimeType() {
  if (typeof MediaRecorder === 'undefined') return '';
  const types = [
    'video/mp4;codecs=h264',
    'video/mp4',
    'video/webm;codecs=vp8',
    'video/webm',
  ];
  if (typeof MediaRecorder.isTypeSupported !== 'function') return '';
  return types.find((type) => MediaRecorder.isTypeSupported(type)) || '';
}

function startVideoRecording(mediaStream) {
  if (!config.video_upload || typeof MediaRecorder === 'undefined') return null;
  const mimeType = preferredVideoMimeType();
  const options = { videoBitsPerSecond: config.video_bits_per_second };
  if (mimeType) options.mimeType = mimeType;
  const recorder = new MediaRecorder(mediaStream, options);
  const chunks = [];
  const stopped = new Promise((resolve, reject) => {
    recorder.addEventListener('dataavailable', (event) => {
      if (event.data.size) chunks.push(event.data);
    });
    recorder.addEventListener('error', () => reject(new Error('Browser video recording failed')));
    recorder.addEventListener('stop', () => {
      resolve(new Blob(chunks, { type: recorder.mimeType || mimeType || 'video/webm' }));
    });
  });
  recorder.start(1000);
  return {
    async stop() {
      if (recorder.state !== 'inactive') recorder.stop();
      return stopped;
    },
  };
}

function waitForSocket(socket, expectedType) {
  return new Promise((resolve, reject) => {
    const handleMessage = (event) => {
      const value = JSON.parse(event.data);
      if (value.type === 'error') { cleanup(); reject(new Error(value.message)); }
      else if (value.type === expectedType) { cleanup(); resolve(value); }
    };
    const handleClose = () => { cleanup(); reject(new Error('Upload connection closed early')); };
    const handleError = () => { cleanup(); reject(new Error('Upload connection failed')); };
    const timer = setTimeout(() => {
      cleanup();
      socket.close();
      reject(new Error('Upload timed out. Check Wi-Fi and try again.'));
    }, config.upload_timeout_seconds * 1000);
    const cleanup = () => {
      clearTimeout(timer);
      socket.removeEventListener('message', handleMessage);
      socket.removeEventListener('close', handleClose);
      socket.removeEventListener('error', handleError);
    };
    socket.addEventListener('message', handleMessage);
    socket.addEventListener('close', handleClose);
    socket.addEventListener('error', handleError);
  });
}

function sendAndWait(socket, payload, expectedType) {
  const response = waitForSocket(socket, expectedType);
  try {
    socket.send(payload);
  } catch (error) {
    socket.close();
    throw error;
  }
  return response;
}

function waitForSocketOpen(socket) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      cleanup();
      socket.close();
      reject(new Error('Could not reach the upload server in time'));
    }, config.upload_timeout_seconds * 1000);
    const cleanup = () => {
      clearTimeout(timer);
      socket.removeEventListener('open', handleOpen);
      socket.removeEventListener('error', handleError);
    };
    const handleOpen = () => { cleanup(); resolve(); };
    const handleError = () => { cleanup(); reject(new Error('Could not connect to upload server')); };
    socket.addEventListener('open', handleOpen, { once: true });
    socket.addEventListener('error', handleError, { once: true });
  });
}

async function uploadFrames(frames, captureId) {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/ws/upload`);
  try {
    await waitForSocketOpen(socket);
    await sendAndWait(socket, JSON.stringify({
      type: 'start', session_id: sessionId, client_id: captureId,
      upload_kind: 'frames', frame_count: frames.length,
    }), 'ready');
    for (let index = 0; index < frames.length; index += 1) {
      detail.textContent = `Uploading ${index + 1} of ${frames.length}…`;
      const payload = await frames[index].blob.arrayBuffer();
      await sendAndWait(socket, payload, 'ack');
    }
    return await sendAndWait(socket, JSON.stringify({ type: 'complete' }), 'complete');
  } finally {
    if (socket.readyState < WebSocket.CLOSING) socket.close();
  }
}

async function uploadVideo(blob, captureId) {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/ws/upload`);
  try {
    await waitForSocketOpen(socket);
    await sendAndWait(socket, JSON.stringify({
      type: 'start', session_id: sessionId, client_id: captureId,
      upload_kind: 'video', mime_type: blob.type,
    }), 'ready');
    detail.textContent = `Uploading ${(blob.size / 1_000_000).toFixed(1)} MB video…`;
    const payload = await blob.arrayBuffer();
    await sendAndWait(socket, payload, 'ack');
    return await sendAndWait(socket, JSON.stringify({ type: 'complete' }), 'complete');
  } finally {
    if (socket.readyState < WebSocket.CLOSING) socket.close();
  }
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
    showToast(`${error.message}<br><br>Camera and motion access require HTTPS.`, 'TRY AGAIN');
    return;
  }
  candidates = [];
  sampleIndex = 0;
  rotationTravel = 0;
  translationSignal = 0;
  lastMotionTime = 0;
  captureStarted = performance.now();
  capturing = true;
  addEventListener('devicemotion', handleMotion);
  updateCaptureWarning();
  startButton.hidden = true;
  count.hidden = false;
  headline.textContent = 'Hold the shared landmark';
  detail.textContent = 'Keep it steady for 2 seconds so every phone has an overlap anchor.';
  let videoRecording = null;
  try {
    videoRecording = startVideoRecording(stream);
  } catch {
    // The rolling JPEG buffer below is the compatibility fallback.
  }
  const durationMs = config.capture_seconds * 1000;
  const sampleEveryMs = 1000 / config.capture_fps;
  let nextSample = 0;

  await new Promise((resolve) => {
    const tick = async (now) => {
      const elapsed = now - captureStarted;
      const progress = Math.min(elapsed / durationMs, 1);
      if (elapsed >= 2500 && elapsed < durationMs - 3000) {
        headline.textContent = 'Walk sideways through your arc';
        detail.textContent = 'Keep nearby surfaces in view; translate instead of panning in place.';
      } else if (elapsed >= durationMs - 3000) {
        headline.textContent = 'Finish on the shared landmark';
        detail.textContent = 'Return to the same corner or object to bridge this phone to the room.';
      }
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
  let recordedVideo = null;
  if (videoRecording) {
    try {
      recordedVideo = await videoRecording.stop();
    } catch {
      recordedVideo = null;
    }
  }
  const sharp = selectTemporalFrames(candidates, config.frames_per_client);
  const passing = sharp.filter((frame) => frame.score >= config.blur_threshold).length;
  const captureId = `${clientId}-${Date.now().toString(36)}`;
  try {
    if (recordedVideo && recordedVideo.size > 0 && recordedVideo.size <= config.max_video_upload_bytes) {
      headline.textContent = 'Sending video to queue';
      await uploadVideo(recordedVideo, captureId);
      showToast(`<strong>Video queued</strong><br>The shared room will refresh after verified alignment.<br><br>Only add another clip for a missing area; begin and end it on the shared landmark.`);
    } else {
      headline.textContent = 'Selecting sharp frames';
      detail.textContent = `${passing}/${sharp.length} frames passed the sharpness target. Uploading fallback frames…`;
      const result = await uploadFrames(sharp, captureId);
      showToast(`<strong>Upload complete</strong><br>${result.frames} sharp frames joined session <b>${sessionId}</b>.<br><br>Scan another low arc for any dark floor or ceiling areas.`);
    }
  } catch (error) {
    showToast(`<strong>Upload failed</strong><br>${error.message}`, 'TRY AGAIN');
  } finally {
    stream?.getTracks().forEach((track) => track.stop());
  }
}

startButton.addEventListener('click', beginCapture);
toastAction.addEventListener('click', resetForAnotherCapture);
