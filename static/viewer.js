import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';

const viewport = document.querySelector('#viewport');
const message = document.querySelector('#message');
const flyButton = document.querySelector('#fly');
const sizeInput = document.querySelector('#point-size');
const mode = document.querySelector('#mode');
const params = new URLSearchParams(location.search);
const session = params.get('session');
const base = session ? `/session/${encodeURIComponent(session)}/` : '/sample_data/';
let configuredPointSize = Number(sizeInput.value);
let maxSplatScreenSize = Number.POSITIVE_INFINITY;
let splatExposure = 1;
let viewerRefreshMs = 2000;
try {
  const runtimeConfig = await fetch('/api/config', { cache: 'no-store' }).then((response) => {
    if (!response.ok) throw new Error('static preview');
    return response.json();
  });
  sizeInput.value = String(runtimeConfig.point_size);
  configuredPointSize = Number(runtimeConfig.point_size);
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
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewport.appendChild(renderer.domElement);
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

function projectedSplatScale() {
  const focalPixels = renderer.domElement.height
    / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
  return focalPixels * Number(sizeInput.value) / Math.max(configuredPointSize, 0.0001);
}

function updateSplatPointScale() {
  if (pointsObject?.material.uniforms?.pointScale) {
    pointsObject.material.uniforms.pointScale.value = projectedSplatScale();
  }
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
  const isSplat = propertyIndex.f_dc_0 !== undefined;
  const splatScales = isSplat ? new Float32Array(count) : null;
  const splatOpacities = isSplat ? new Float32Array(count) : null;
  for (let index = 0; index < count; index += 1) {
    const values = rows[index].trim().split(/\s+/).map(Number);
    positions.set([values[propertyIndex.x], values[propertyIndex.y], values[propertyIndex.z]], index * 3);
    if (isSplat) {
      colors.set([
        THREE.MathUtils.clamp(0.5 + 0.2820947918 * values[propertyIndex.f_dc_0], 0, 1),
        THREE.MathUtils.clamp(0.5 + 0.2820947918 * values[propertyIndex.f_dc_1], 0, 1),
        THREE.MathUtils.clamp(0.5 + 0.2820947918 * values[propertyIndex.f_dc_2], 0, 1),
      ], index * 3);
      splatScales[index] = Math.exp((values[propertyIndex.scale_0] + values[propertyIndex.scale_1] + values[propertyIndex.scale_2]) / 3);
      splatOpacities[index] = 1 / (1 + Math.exp(-values[propertyIndex.opacity]));
    } else {
      colors.set([
        values[propertyIndex.red] / 255,
        values[propertyIndex.green] / 255,
        values[propertyIndex.blue] / 255,
      ], index * 3);
    }
  }
  return { positions, colors, isSplat, splatScales, splatOpacities };
}

function splatMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      pointScale: { value: projectedSplatScale() },
      maxPointSize: { value: maxSplatScreenSize },
      exposure: { value: splatExposure },
    },
    // Keep the custom splat color attribute distinct from Three's built-in
    // `color` attribute, which the renderer may inject into shader programs.
    vertexColors: false,
    transparent: true,
    depthWrite: false,
    vertexShader: `
      attribute float splatScale;
      attribute float splatOpacity;
      attribute vec3 splatColor;
      varying vec3 vColor;
      varying float vOpacity;
      uniform float pointScale;
      uniform float maxPointSize;
      void main() {
        vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * viewPosition;
        gl_PointSize = clamp(
          pointScale * splatScale / max(-viewPosition.z, 0.01),
          1.0,
          maxPointSize
        );
        vColor = splatColor;
        vOpacity = splatOpacity;
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      varying float vOpacity;
      uniform float exposure;
      void main() {
        vec2 centered = gl_PointCoord - vec2(0.5);
        float radius2 = dot(centered, centered);
        if (radius2 > 0.25) discard;
        float gaussian = exp(-radius2 * 18.0);
        gl_FragColor = vec4(min(vColor * exposure, vec3(1.0)), gaussian * vOpacity);
      }
    `,
  });
}

async function loadCloud(version = '', modelPath = '') {
  const suffix = version ? `?v=${encodeURIComponent(version)}` : '';
  const revisionBase = `${base}${modelPath}`;
  let response = await fetch(`${revisionBase}splat.ply${suffix}`, { cache: 'no-store' });
  if (!response.ok) response = await fetch(`${revisionBase}points.ply${suffix}`, { cache: 'no-store' });
  if (!response.ok) throw new Error('No reconstruction is available for this session');
  let parsed;
  try {
    parsed = parseAsciiPly(await response.text());
  } catch (error) {
    // A splat format the lightweight viewer cannot read must not prevent the
    // point-cloud safety artifact from loading.
    const fallback = await fetch(`${revisionBase}points.ply${suffix}`, { cache: 'no-store' });
    if (!fallback.ok) throw error;
    parsed = parseAsciiPly(await fallback.text());
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(parsed.positions, 3));
  if (parsed.isSplat) {
    geometry.setAttribute('splatColor', new THREE.BufferAttribute(parsed.colors, 3));
    geometry.setAttribute('splatScale', new THREE.BufferAttribute(parsed.splatScales, 1));
    geometry.setAttribute('splatOpacity', new THREE.BufferAttribute(parsed.splatOpacities, 1));
  } else {
    geometry.setAttribute('color', new THREE.BufferAttribute(parsed.colors, 3));
  }
  geometry.computeBoundingSphere();
  const material = parsed.isSplat ? splatMaterial() : new THREE.PointsMaterial({
    size: Number(sizeInput.value),
    vertexColors: true,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.98,
  });
  material.size = Number(sizeInput.value);
  return new THREE.Points(geometry, material);
}

async function loadFlight(version = '', modelPath = '') {
  const suffix = version ? `?v=${encodeURIComponent(version)}` : '';
  const response = await fetch(`${base}${modelPath}cameras.json${suffix}`, { cache: 'no-store' });
  if (!response.ok) throw new Error('Camera path is unavailable');
  const cameras = await response.json();
  const referenceK = cameras[0]?.K;
  const focalY = Number(referenceK?.[1]?.[1]);
  const principalY = Number(referenceK?.[1]?.[2]);
  let recoveredFov;
  if (focalY > 0 && principalY > 0) {
    // K uses image pixels. VGGT centers the principal point, so 2*cy is the
    // recovered image height and gives the matching vertical field of view.
    recoveredFov = THREE.MathUtils.radToDeg(2 * Math.atan(principalY / focalY));
  }
  const recoveredFlight = cameras.map((entry) => {
    const values = entry.T_wc.flat();
    const matrix = new THREE.Matrix4().set(...values).multiply(cvToThree);
    return {
      position: new THREE.Vector3().setFromMatrixPosition(matrix),
      quaternion: new THREE.Quaternion().setFromRotationMatrix(matrix),
    };
  });
  return { recoveredFlight, recoveredFov };
}

function installModel(nextPoints, nextFlight) {
  const previous = pointsObject;
  pointsObject = nextPoints;
  scene.add(pointsObject);
  if (previous) {
    scene.remove(previous);
    previous.geometry.dispose();
    previous.material.dispose();
  }
  const sphere = pointsObject.geometry.boundingSphere;
  controls.target.copy(sphere.center);
  camera.position.set(
    sphere.center.x,
    sphere.center.y + sphere.radius * 0.35,
    sphere.center.z + sphere.radius * 1.15,
  );
  if (nextFlight.recoveredFov) camera.fov = nextFlight.recoveredFov;
  camera.updateProjectionMatrix();
  flight = nextFlight.recoveredFlight;
  flying = flight.length >= 2;
  controls.enabled = !flying;
  flyButton.textContent = flying ? 'Pause flythrough' : 'Resume flythrough';
  mode.textContent = flying ? 'Recovered-camera flythrough' : 'Free orbit';
  controls.update();
  updateSplatPointScale();
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
  const index = Math.floor(elapsed) % flight.length;
  const next = (index + 1) % flight.length;
  const amount = smoothstep(elapsed - Math.floor(elapsed));
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
  if (!pointsObject) return;
  if (pointsObject.material.uniforms?.pointScale) {
    updateSplatPointScale();
  } else {
    pointsObject.material.size = Number(sizeInput.value);
  }
});
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  updateSplatPointScale();
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
