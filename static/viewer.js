import * as THREE from 'three';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import { OrbitControls } from '/static/vendor/OrbitControls.js';

const viewport = document.querySelector('#viewport');
const message = document.querySelector('#message');
const flyButton = document.querySelector('#fly');
const sizeInput = document.querySelector('#point-size');
const sizeControl = document.querySelector('#point-size-control');
const mode = document.querySelector('#mode');
const params = new URLSearchParams(location.search);
const session = params.get('session');
const base = session ? `/session/${encodeURIComponent(session)}/` : '/sample_data/';
let maxSplatScreenSize = 512;
let splatExposure = 1;
let viewerRefreshMs = 2000;
try {
  const runtimeConfig = await fetch('/api/config', { cache: 'no-store' }).then((response) => {
    if (!response.ok) throw new Error('static preview');
    return response.json();
  });
  sizeInput.value = String(runtimeConfig.point_size);
  maxSplatScreenSize = Number(runtimeConfig.splat_max_screen_size);
  splatExposure = Number(runtimeConfig.splat_exposure);
  viewerRefreshMs = Number(runtimeConfig.viewer_refresh_seconds) * 1000;
} catch {
  // Standalone sample_data previews use the value embedded in viewer.html.
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07090d);
scene.fog = new THREE.FogExp2(0x07090d, 0.012);
const camera = new THREE.PerspectiveCamera(62, innerWidth / innerHeight, 0.01, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewport.appendChild(renderer.domElement);
const spark = new SparkRenderer({
  renderer,
  enableLod: true,
  maxStdDev: Math.sqrt(8),
  maxPixelRadius: maxSplatScreenSize,
});
scene.add(spark);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;

let pointsObject;
let flight = [];
let flying = true;
let flightStart = performance.now();
let modelVersion = '';
let modelLoading = false;
const segmentMs = 1350;
const cvToThree = new THREE.Matrix4().makeScale(1, -1, -1);

function estimateRecoveredUp(matrices) {
  const candidates = matrices.map((matrix) => (
    new THREE.Vector3().setFromMatrixColumn(matrix, 1).normalize()
  ));
  if (!candidates.length) return new THREE.Vector3(0, 1, 0);

  // Pick the camera-up axis with the strongest agreement, then align the
  // remaining axes to it before averaging. This rejects an occasional 180°
  // pose-roll error without assuming that model-space +Y is room-space up.
  const reference = candidates.reduce((best, candidate) => {
    const score = candidates.reduce((sum, other) => sum + Math.abs(candidate.dot(other)), 0);
    return score > best.score ? { axis: candidate, score } : best;
  }, { axis: candidates[0], score: -Infinity }).axis;
  const recoveredUp = new THREE.Vector3();
  candidates.forEach((candidate) => {
    recoveredUp.addScaledVector(candidate, candidate.dot(reference) < 0 ? -1 : 1);
  });
  return recoveredUp.lengthSq() > 1e-8 ? recoveredUp.normalize() : reference.clone();
}

function leveledCameraPose(matrix, recoveredUp) {
  const position = new THREE.Vector3().setFromMatrixPosition(matrix);
  const forward = new THREE.Vector3(0, 0, -1).transformDirection(matrix);
  const original = new THREE.Quaternion().setFromRotationMatrix(matrix);
  if (Math.abs(forward.dot(recoveredUp)) > 0.985) {
    return { position, quaternion: original };
  }
  const rotation = new THREE.Matrix4().lookAt(
    position,
    position.clone().add(forward),
    recoveredUp,
  );
  return {
    position,
    quaternion: new THREE.Quaternion().setFromRotationMatrix(rotation),
  };
}

function parseAsciiPly(text) {
  const end = text.indexOf('end_header');
  if (end < 0) throw new Error('PLY header is incomplete');
  const header = text.slice(0, end).split(/\r?\n/);
  if (!header.some((line) => line.trim() === 'format ascii 1.0')) throw new Error('Only ASCII PLY is supported');
  const vertexLine = header.find((line) => line.startsWith('element vertex '));
  const count = Number(vertexLine?.split(/\s+/)[2]);
  const properties = [];
  let inVertex = false;
  for (const line of header) {
    if (line.startsWith('element ')) inVertex = line.startsWith('element vertex ');
    else if (inVertex && line.startsWith('property ')) properties.push(line.trim().split(/\s+/).at(-1));
  }
  const propertyIndex = Object.fromEntries(properties.map((name, index) => [name, index]));
  const rows = text.slice(end + 'end_header'.length).trim().split(/\r?\n/);
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  for (let index = 0; index < count; index += 1) {
    const values = rows[index].trim().split(/\s+/).map(Number);
    positions.set([values[propertyIndex.x], values[propertyIndex.y], values[propertyIndex.z]], index * 3);
    colors.set([
      values[propertyIndex.red] / 255,
      values[propertyIndex.green] / 255,
      values[propertyIndex.blue] / 255,
    ], index * 3);
  }
  return { positions, colors };
}

async function loadSparkSplat(url) {
  const available = await fetch(url, { method: 'HEAD', cache: 'no-store' });
  if (!available.ok) return null;
  let splat;
  try {
    splat = new SplatMesh({
      url,
      fileType: 'ply',
      fileName: 'splat.ply',
      lod: true,
      // Spark keeps only its generated LoD tree by default. Retain the compact
      // base array as well so getBoundingBox() can frame the loaded scene.
      nonLod: true,
    });
    await splat.initialized;
    splat.recolor.setRGB(splatExposure, splatExposure, splatExposure);
    const sphere = splat.getBoundingBox(true).getBoundingSphere(new THREE.Sphere());
    if (!Number.isFinite(sphere.radius) || sphere.radius <= 0) {
      throw new Error('Splat has no finite bounds');
    }
    splat.userData.roomscanKind = 'splat';
    splat.userData.roomscanBoundingSphere = sphere;
    return splat;
  } catch {
    splat?.dispose();
    return null;
  }
}

async function loadCloud(version = '', modelPath = '') {
  const suffix = version ? `?v=${encodeURIComponent(version)}` : '';
  const revisionBase = `${base}${modelPath}`;
  const splat = await loadSparkSplat(`${revisionBase}splat.ply${suffix}`);
  if (splat) return splat;
  const response = await fetch(`${revisionBase}points.ply${suffix}`, { cache: 'no-store' });
  if (!response.ok) throw new Error('No reconstruction is available for this session');
  const parsed = parseAsciiPly(await response.text());
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(parsed.positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(parsed.colors, 3));
  geometry.computeBoundingSphere();
  const material = new THREE.PointsMaterial({
    size: Number(sizeInput.value),
    vertexColors: true,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.98,
  });
  const points = new THREE.Points(geometry, material);
  points.userData.roomscanKind = 'points';
  points.userData.roomscanBoundingSphere = geometry.boundingSphere;
  return points;
}

async function loadFlight(version = '', modelPath = '') {
  const suffix = version ? `?v=${encodeURIComponent(version)}` : '';
  const response = await fetch(`${base}${modelPath}cameras.json${suffix}`, { cache: 'no-store' });
  if (!response.ok) throw new Error('Camera path is unavailable');
  const cameras = await response.json();
  const paths = new Map();
  cameras.forEach((entry, index) => {
    let stem = String(entry.frame ?? index).split('/').at(-1).replace(/\.[^.]+$/, '');
    while (/^\d{3}_.+_\d+$/.test(stem)) stem = stem.slice(4);
    const capture = stem.match(/^(.*)_\d+$/)?.[1] ?? '__sequence__';
    if (!paths.has(capture)) paths.set(capture, []);
    paths.get(capture).push(entry);
  });
  // Show one coherent recovered clip. Even aligned clips must not be joined by
  // an invented camera segment that passes through unobserved space.
  const flightCameras = [...paths.values()].sort((left, right) => right.length - left.length)[0] ?? [];
  const referenceK = flightCameras[0]?.K;
  const focalY = Number(referenceK?.[1]?.[1]);
  const principalY = Number(referenceK?.[1]?.[2]);
  let recoveredFov;
  if (focalY > 0 && principalY > 0) {
    // K uses image pixels. VGGT centers the principal point, so 2*cy is the
    // recovered image height and gives the matching vertical field of view.
    recoveredFov = THREE.MathUtils.radToDeg(2 * Math.atan(principalY / focalY));
  }
  const recoveredMatrices = cameras.map((entry) => {
    const values = entry.T_wc.flat();
    return new THREE.Matrix4().set(...values).multiply(cvToThree);
  });
  const recoveredUp = estimateRecoveredUp(recoveredMatrices);
  const recoveredFlight = flightCameras.map((entry) => {
    const values = entry.T_wc.flat();
    const matrix = new THREE.Matrix4().set(...values).multiply(cvToThree);
    return leveledCameraPose(matrix, recoveredUp);
  });
  return { recoveredFlight, recoveredFov, recoveredUp };
}

function installModel(nextPoints, nextFlight) {
  const previous = pointsObject;
  pointsObject = nextPoints;
  scene.add(pointsObject);
  if (previous) {
    scene.remove(previous);
    if (previous.userData.roomscanKind === 'splat') {
      previous.dispose();
    } else {
      previous.geometry.dispose();
      previous.material.dispose();
    }
  }
  const isSplat = pointsObject.userData.roomscanKind === 'splat';
  sizeControl.hidden = isSplat;
  const sphere = pointsObject.userData.roomscanBoundingSphere;
  const recoveredUp = nextFlight.recoveredUp ?? new THREE.Vector3(0, 1, 0);
  camera.up.copy(recoveredUp);
  controls.target.copy(sphere.center);
  const orbitDirection = new THREE.Vector3(0, 0, 1)
    .addScaledVector(recoveredUp, -recoveredUp.z);
  if (orbitDirection.lengthSq() < 1e-8) orbitDirection.set(1, 0, 0);
  orbitDirection.normalize();
  camera.position.copy(sphere.center)
    .addScaledVector(recoveredUp, sphere.radius * 0.35)
    .addScaledVector(orbitDirection, sphere.radius * 1.15);
  if (nextFlight.recoveredFov) camera.fov = nextFlight.recoveredFov;
  camera.updateProjectionMatrix();
  flight = nextFlight.recoveredFlight;
  flying = flight.length >= 2;
  controls.enabled = !flying;
  flyButton.textContent = flying ? 'Pause flythrough' : 'Resume flythrough';
  mode.textContent = flying ? 'Recovered-camera flythrough' : 'Free orbit';
  controls.update();
  flightStart = performance.now();
}

async function reloadModel(version = '', modelPath = '') {
  if (modelLoading) return;
  modelLoading = true;
  if (pointsObject) {
    message.textContent = 'Updating shared reconstruction…';
    message.hidden = false;
  }
  try {
    const [nextPoints, nextFlight] = await Promise.all([
      loadCloud(version, modelPath), loadFlight(version, modelPath),
    ]);
    installModel(nextPoints, nextFlight);
    message.hidden = true;
    if (version) modelVersion = String(version);
  } finally {
    modelLoading = false;
  }
}

async function pollForModelUpdate() {
  if (!session) return;
  try {
    const response = await fetch(`/api/session/${encodeURIComponent(session)}/status`, {
      cache: 'no-store',
    });
    if (!response.ok) return;
    const status = await response.json();
    const nextVersion = String(status.model_version || '');
    if (status.viewer_ready && nextVersion && nextVersion !== modelVersion) {
      await reloadModel(nextVersion, String(status.model_path || ''));
    } else if (status.job.status === 'running' && pointsObject) {
      message.textContent = 'Updating shared reconstruction…';
      message.hidden = false;
    } else if (pointsObject) {
      message.hidden = true;
    }
  } catch {
    // Keep the last good model visible through transient status failures.
  }
}

function smoothstep(value) { return value * value * (3 - 2 * value); }

function updateFlight(now) {
  if (!flying || flight.length < 2) return;
  // The first rAF timestamp can predate flightStart by a fraction of a frame.
  // Clamp it so JavaScript's negative remainder cannot produce index -1.
  const elapsed = Math.max(0, (now - flightStart) / segmentMs);
  const segment = Math.floor(elapsed);
  const lastIndex = flight.length - 1;
  const cycleSegment = segment % (lastIndex * 2);
  const forward = cycleSegment < lastIndex;
  const index = forward ? cycleSegment : lastIndex - (cycleSegment - lastIndex);
  const next = forward ? index + 1 : index - 1;
  const amount = smoothstep(elapsed - segment);
  // Segment interpolation stays inside the recovered-camera hull: the path
  // cannot overshoot outside capture volume as a spline can.
  camera.position.lerpVectors(flight[index].position, flight[next].position, amount);
  camera.quaternion.slerpQuaternions(flight[index].quaternion, flight[next].quaternion, amount);
}

function animate(now) {
  requestAnimationFrame(animate);
  updateFlight(now);
  if (!flying) controls.update();
  renderer.render(scene, camera);
}

flyButton.addEventListener('click', () => {
  flying = !flying;
  controls.enabled = !flying;
  flyButton.textContent = flying ? 'Pause flythrough' : 'Resume flythrough';
  mode.textContent = flying ? 'Recovered-camera flythrough' : 'Free orbit';
  if (flying) flightStart = performance.now();
});
renderer.domElement.addEventListener('pointerdown', () => {
  if (flying) flyButton.click();
}, { passive: true });
sizeInput.addEventListener('input', () => {
  if (pointsObject?.userData.roomscanKind === 'points') {
    pointsObject.material.size = Number(sizeInput.value);
  }
});
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

try {
  if (session) {
    const status = await fetch(`/api/session/${encodeURIComponent(session)}/status`, {
      cache: 'no-store',
    }).then((response) => response.json());
    const initialVersion = String(status.model_version || '');
    await reloadModel(initialVersion, String(status.model_path || ''));
  } else {
    await reloadModel();
  }
} catch (error) {
  message.textContent = error.message;
  flying = false;
  controls.enabled = true;
}
if (session) setInterval(pollForModelUpdate, viewerRefreshMs);
requestAnimationFrame(animate);
