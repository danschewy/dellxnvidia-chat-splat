# ROOMSCAN

The hackathon pitch deck is a self-contained, keyboard-navigable HTML presentation at [`docs/pitch/ROOMSCAN-Pitch-Deck.html`](docs/pitch/ROOMSCAN-Pitch-Deck.html). Open it in a browser; use the arrow keys to present, `F` for fullscreen, or `P` to print to PDF.

## Demo

<video controls playsinline preload="metadata" poster="docs/pitch/roomscan-live.png" width="100%">
  <source src="docs/pitch/roomscan-demo.mp4" type="video/mp4">
  Your Markdown viewer cannot play embedded video. Use the direct link below.
</video>

[▶ Play or download the ROOMSCAN demo](docs/pitch/roomscan-demo.mp4)

ROOMSCAN turns frames from several phones into a shared colored point cloud and, when CUDA/gsplat is healthy, an optional Gaussian splat. The browser renders real splats with the locally vendored SparkJS renderer and falls back to Three.js points when no splat exists. It starts with a flythrough along one coherent recovered clip and reverses at its endpoints, so it never invents a path through empty space between phones.

The system is deliberately usable without an ML environment: `StubBackend` reconstructs the complete fixture scene in a fraction of a second. CUDA, MPS, and stub all write the same disk contract.

## Output contract

Each scan lives under one directory:

```text
session/<id>/
  uploads/<client>.mp4 # original phone video (webm/mov also accepted)
  frames/*.jpg
  points.ply
  cameras.json
  splat.ply          # optional
  meta.json
  current.json       # atomic pointer to the active immutable revision
  models/<version>/  # matching points/cameras/meta generation
```

Frames are written before masking; masked/resized derivatives live under `work/`. `points.ply` and `cameras.json` are atomically written before splat training begins. A gsplat failure is recorded in `meta.json` and does not remove them.

Phone capture is video-first. The browser records one 15-second clip and sends it as one WebSocket binary upload. The server persists that original under `uploads/`, queues OpenCV/FFmpeg decoding in the background, samples at `capture_fps`, divides the clip into temporal windows, and keeps the sharpest frame in each window. That preserves bridge views instead of selecting twenty similar sharp frames from one moment. Browsers without a usable `MediaRecorder` automatically fall back to the original client-side JPEG flow with the same temporal selection policy.

The server root is the room lobby. **New room scan** creates a filesystem-backed session and opens its capture page; a room code or full invite URL joins an existing session. Existing-room cards are newest-first and use the latest captured frame as their thumbnail. Direct QR links keep the stable `/?session=<room-id>` capture contract, and **View room** opens the shared reconstruction when one has been published.

With `live_updates` enabled, every completed capture marks its session dirty. ROOMSCAN waits `live_update_debounce_seconds` for nearby clips to batch together, but `live_update_max_wait_seconds` forces a rebuild even when clips keep arriving. A clip completed during reconstruction schedules exactly one follow-up pass. Each model builds in a private staging directory and is published as an immutable revision; `current.json` atomically switches the viewer to matching point-cloud and camera files. Open viewers poll `model_version` and swap to the new cloud and camera path without a page reload. `live_update_train_splat` defaults off so the validated Pi3X geometry reaches viewers as soon as it is ready. Enable it only after the splat trainer passes its held-out camera quality gate on the target room; a rejected splat never replaces the durable point cloud.

Automatic publication also checks camera continuity inside each phone clip. A clearly discontinuous rebuild cannot replace an already-published revision that passed the check: the status page reports `HELD`, the last good shared model remains visible, and a new clip can trigger another attempt. First models and explicit **Rebuild now** requests are never blocked by this heuristic.

The process keeps one shared backend instance, so VGGT and YOLO weights remain loaded across live revisions. A bounded global inference queue serializes reconstructions across sessions on the single GPU. Video decoding remains separately bounded by `video_worker_count`; uploaded video is capped by bytes, duration, source FPS, and sampled-frame count, JPEG fallback is capped by frame count, and stalled WebSockets time out. Only `model_revision_retention` immutable model generations are retained per session (three by default), including the active revision and a rollback window.

The root-level files remain for CLI/offline compatibility, but a live reader must resolve `current.json` and read the matching immutable `models/<version>/` directory. That is the transaction boundary used by the browser; reading root `points.ply` and `cameras.json` concurrently with publication is not guaranteed to return the same generation.

## Mac setup (Apple Silicon)

Use Python 3.11 or 3.12; some ML wheels may lag newer Python releases.

```bash
cd roomscan
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mac.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

For frontend/server work, force the instant fixture backend:

```bash
ROOMSCAN_BACKEND=stub python check_env.py
ROOMSCAN_BACKEND=stub python reconstruct.py sample_data/frames /tmp/roomscan-demo
ROOMSCAN_BACKEND=stub python server.py --self-signed
```

Open `https://<mac-lan-ip>:8443/?session=demo`. The first visit uses a development certificate, so each phone must explicitly trust/continue past the browser warning. MPS is selected automatically when a compatible PyTorch build is installed, but MPS output is only a smoke test; it is not CUDA validation. `train_splat()` intentionally returns `None` on MPS.

## GB10 / DGX OS setup

The box must have an NVIDIA-supported arm64 PyTorch build whose CUDA runtime matches its driver. Preserve an existing NVIDIA build when present—do not replace it with generic PyPI torch. On a clean CUDA 13 DGX OS install with no system torch, this is the combination validated for ROOMSCAN:

```bash
cd roomscan
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install 'ninja>=1.11,<2'
CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH" MAX_JOBS=4 \
  python -m pip install --no-build-isolation -r requirements-cuda.txt
CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH" \
  ROOMSCAN_WEIGHTS_DIR=/absolute/path/to/hackathon-models python check_env.py
```

If DGX OS already supplies a working CUDA torch, create the environment with `--system-site-packages`, omit the torch installation line, and verify its version before installing anything else. If the image's bundled pip 24 repeatedly fails while parsing the package index, bootstrap a current pip, then repeat the packaging command:

```bash
curl -fsSLo /tmp/roomscan-get-pip.py https://bootstrap.pypa.io/get-pip.py
python /tmp/roomscan-get-pip.py
```

`check_env.py` forces the gsplat CUDA module to load/build, rather than merely importing gsplat's lazy Python wrapper. On CUDA, a failure is printed as a loud M0 failure. Do not proceed to a live demo until the four-frame VGGT forward pass, YOLO load, and compiled gsplat module all report `OK`.

The reference box run was verified on NVIDIA GB10, driver 580.159.03, CUDA toolkit 13.0.88, Python 3.12.3, and PyTorch 2.11.0+cu130. The 25-frame CUDA fixture produced 3,807,300 points and a 100,000-Gaussian splat in 34.03 seconds total. A separate 18-photo occupied-room run used the production 300,000-splat budget, removed 882,493 masked person pixels before geometry filtering, retained 1,386,310 room points, and completed in 33.78 seconds. The final 20-frame portrait-video run retained a bounded 1,000,000-point cloud, trained 300,000 initialized Gaussians for 3,000 iterations, and completed in 42.69 seconds with the 518×616 balanced input. Its Spark-rendered recovered-camera flythrough was visually verified in Chrome with no console errors.

Run the production server with certificate paths supplied by the environment:

```bash
export ROOMSCAN_WEIGHTS_DIR=/absolute/path/to/hackathon-models
export ROOMSCAN_CERT_FILE=/absolute/path/to/fullchain.pem
export ROOMSCAN_KEY_FILE=/absolute/path/to/privkey.pem
source .venv/bin/activate
python server.py --host 0.0.0.0 --port 8443
```

No runtime downloads are allowed. `models.py` enables Hugging Face/Transformers offline modes and resolves only local paths.
The viewer is offline too: Three.js and SparkJS 2.1.0 are committed under `static/vendor/`; it never imports a CDN module at runtime.

## Pre-download weights on a connected staging machine

Build the exact directory structure expected by `models.py`:

```bash
export WEIGHTS=/absolute/path/to/hackathon-models
mkdir -p "$WEIGHTS/meta/VGGT" "$WEIGHTS/meta/Pi3" "$WEIGHTS/meta/Pi3X" "$WEIGHTS/yolo"
hf download facebook/VGGT-1B --include config.json --include model.safetensors \
  --local-dir "$WEIGHTS/meta/VGGT"
hf download yyfz233/Pi3 --include config.json --include model.safetensors \
  --local-dir "$WEIGHTS/meta/Pi3"
hf download yyfz233/Pi3X --include config.json --include model.safetensors \
  --local-dir "$WEIGHTS/meta/Pi3X"
curl -fL --retry 3 \
  -o "$WEIGHTS/yolo/yolo11s-seg.pt" \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s-seg.pt
hf cache verify facebook/VGGT-1B --local-dir "$WEIGHTS/meta/VGGT"
hf cache verify yyfz233/Pi3 --local-dir "$WEIGHTS/meta/Pi3"
hf cache verify yyfz233/Pi3X --local-dir "$WEIGHTS/meta/Pi3X"
```

Clone and install the VGGT Python package during setup (the requirements files do this); its source code is separate from the weight bundle. Copy the completed virtual environment/source installation and weights to the offline box, or repeat installation while the box is still connected. Then disconnect networking and run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ROOMSCAN_WEIGHTS_DIR="$WEIGHTS" python check_env.py
```

A missing model error always names the exact expected local path.

Pi3X is the production geometry model for unordered crowd captures; its smoother point head and permutation-equivariant camera frame behaved better on the four-phone room test. Set `geometry_model` to `vggt` for a single coherent clip or `pi3` for the older cached fallback. On CUDA, a non-OOM VGGT runtime failure can still activate Pi3 when its local weights are present; OOM follows the frame-subsampling path first.

## When gsplat will not build

Do this at M0, not during the demo:

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name())"
nvcc --version
python -m pip install 'ninja>=1.11,<2'
MAX_JOBS=4 python -m pip install --no-build-isolation --force-reinstall 'gsplat>=1.5,<2'
python -c "from gsplat.cuda._backend import _C; assert _C is not None; print(_C)"
python check_env.py
```

If PyTorch CUDA, the installed toolkit, and the driver are incompatible, fix that environment rather than hiding the error. If gsplat still fails near demo time, set `ROOMSCAN_BACKEND=stub` for the guaranteed fixture demo, or leave CUDA selected and use `points.ply`: splat failure is non-fatal and the viewer automatically falls back to the point cloud. Never try to install gsplat on the Mac.

## Offline reconstruction

```bash
python reconstruct.py /path/to/images /path/to/output
```

VGGT input is prepared once at its native 518-pixel width and then loaded directly without a second resize or crop. The production `crop` mode follows [Meta's official VGGT loader](https://github.com/facebookresearch/vggt/blob/main/vggt/utils/load_fn.py) for width and patch alignment, but raises the configurable portrait cap to `vggt_portrait_height` (616 pixels on the GB10) instead of forcing every portrait frame to 518×518. On the 720×1280 test clip this preserves about 67% of the vertical field rather than 56%, while remaining inside the latency budget. `pad` preserves every pixel but reduces portrait horizontal content to roughly 294 pixels and was materially worse on the test video. Landscape capture remains preferred because it preserves the complete vertical field without padding.

At most 32 evenly spaced frames are used by default. A CUDA/MPS OOM halves the frame set and retries down to four frames. `confidence_threshold` is a 0–1 rejected percentile because VGGT confidence is not normalized. VGGT's confidence activation has an exact floor of `1.0`; ROOMSCAN preserves every point above the percentile boundary, then fills the remaining quota by sampling evenly across tied boundary values. `max_point_cloud_points` bounds that quota at one million by default. This avoids both extremes seen in difficult phone footage: retaining every one of millions of uncertain floor pixels, or retaining only thin high-confidence edges and losing wall coverage. An all-equal stub/model output is preserved unless it exceeds the configured cap. When people masking is enabled, the dilated image masks also exclude the corresponding dense 3D points before confidence filtering and splat initialization.

Every stage logs wall-clock seconds and updates `meta.json`. All operational knobs—including masking, frame caps, upload limits, motion heuristic, and splat training—live in `config.json`.

## Live success test

1. Start the HTTPS server on the box and open `/status?session=demo` on the presentation browser.
2. Put `https://<box-host>:8443/?session=demo` in a QR code.
3. Choose one distinctive shared landmark visible from much of the room: a poster, doorway corner, or high-contrast object. Tell every participant to hold the phone sideways, press **START**, hold that landmark in view for the first 2–3 seconds, then walk sideways in a short arc while filming nearby surfaces. They should return to the landmark for the final 3 seconds. A red warning appears for portrait capture or when motion looks like rotation without translation.
4. Watch clips enter the video queue and become ready at roughly 20 selected frames per phone. Nearby completions are batched; continuous arrivals force a snapshot after the configured maximum wait.
5. Watch stage times in stdout. Open the viewer as soon as `viewer_ready` becomes true; it refreshes to each safely published revision automatically. **Rebuild now** remains available as a manual fallback when the queue is idle.
6. The viewer should show a recognizable occupied room within 90 seconds of the last upload, with people removed, and start on the recovered-camera flythrough. Use free orbit only as a secondary check.

The acceptance run must be performed on CUDA. Stub verification proves the capture/server/viewer contract, not reconstruction quality. MPS verification proves only that imports and a small forward pass can execute.

The 15 seconds is a per-clip latency budget, not a session limit. A phone may submit another clip immediately; every capture gets a unique client/capture ID and joins the shared reconstruction. Stand roughly 1–3 meters from surfaces, keep them visible across several overlapping angles, and translate sideways rather than panning from one spot. Do not walk straight toward a wall, whip-pan, zoom, or cover the lens. Black or missing regions mean there was no confident geometry there: record the next clip facing that region, but start and finish on the shared landmark so ROOMSCAN can align it to the existing room. For floor coverage, use a second low sideways arc aimed slightly downward; bags, chairs, and people must be seen around from another angle because masking cannot invent the floor behind them.

## Tests

```bash
ROOMSCAN_BACKEND=stub python -m unittest discover -s tests -v
```

The suite checks fixture validity, backend selection without ML dependencies, one-pass portrait preprocessing, temporal video selection, camera-continuity quality gating, durable late-stage failure behavior, OOM subsampling, local-only viewer assets, WebSocket upload placement, manual triggering, immutable publication, and the complete stub pipeline.
