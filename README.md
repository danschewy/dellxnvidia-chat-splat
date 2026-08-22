# ROOMSCAN

ROOMSCAN turns frames from several phones into a shared colored point cloud and, when CUDA/gsplat is healthy, an optional Gaussian splat. The browser viewer starts with a flythrough that stays on line segments between recovered camera poses, so it never invents a path outside the capture volume.

The system is deliberately usable without an ML environment: `StubBackend` reconstructs the complete fixture scene in a fraction of a second. CUDA, MPS, and stub all write the same disk contract.

## Output contract

Each scan lives under one directory:

```text
session/<id>/
  frames/*.jpg
  points.ply
  cameras.json
  splat.ply          # optional
  meta.json
```

Frames are written before masking; masked/resized derivatives live under `work/`. `points.ply` and `cameras.json` are atomically written before splat training begins. A gsplat failure is recorded in `meta.json` and does not remove them.

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

The reference box run was verified on NVIDIA GB10, driver 580.159.03, CUDA toolkit 13.0.88, Python 3.12.3, and PyTorch 2.11.0+cu130. The 25-frame CUDA fixture produced 3,807,300 points and a 100,000-Gaussian splat in 34.03 seconds total.

Run the production server with certificate paths supplied by the environment:

```bash
export ROOMSCAN_WEIGHTS_DIR=/absolute/path/to/hackathon-models
export ROOMSCAN_CERT_FILE=/absolute/path/to/fullchain.pem
export ROOMSCAN_KEY_FILE=/absolute/path/to/privkey.pem
.venv/bin/python server.py --host 0.0.0.0 --port 8443
```

No runtime downloads are allowed. `models.py` enables Hugging Face/Transformers offline modes and resolves only local paths.

## Pre-download weights on a connected staging machine

Build the exact directory structure expected by `models.py`:

```bash
export WEIGHTS=/absolute/path/to/hackathon-models
mkdir -p "$WEIGHTS/meta/VGGT" "$WEIGHTS/meta/Pi3" "$WEIGHTS/yolo"
hf download facebook/VGGT-1B --include config.json --include model.safetensors \
  --local-dir "$WEIGHTS/meta/VGGT"
hf download yyfz233/Pi3 --include config.json --include model.safetensors \
  --local-dir "$WEIGHTS/meta/Pi3"
curl -fL --retry 3 \
  -o "$WEIGHTS/yolo/yolo11s-seg.pt" \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s-seg.pt
hf cache verify facebook/VGGT-1B --local-dir "$WEIGHTS/meta/VGGT"
hf cache verify yyfz233/Pi3 --local-dir "$WEIGHTS/meta/Pi3"
```

Clone and install the VGGT Python package during setup (the requirements files do this); its source code is separate from the weight bundle. Copy the completed virtual environment/source installation and weights to the offline box, or repeat installation while the box is still connected. Then disconnect networking and run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ROOMSCAN_WEIGHTS_DIR="$WEIGHTS" python check_env.py
```

A missing model error always names the exact expected local path.

VGGT is the default geometry model. Set `geometry_model` to `pi3` in `config.json` to force the permutation-equivariant fallback when input ordering is suspect. On CUDA, a non-OOM VGGT runtime failure also activates Pi3 automatically when its local weights are present; OOM still follows the frame-subsampling path first.

## When gsplat will not build

Do this at M0, not during the demo:

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name())"
nvcc --version
MAX_JOBS=4 pip install --no-build-isolation --force-reinstall 'gsplat>=1.5,<2'
python -c "from gsplat.cuda._backend import _C; assert _C is not None; print(_C)"
python check_env.py
```

If PyTorch CUDA, the installed toolkit, and the driver are incompatible, fix that environment rather than hiding the error. If gsplat still fails near demo time, set `ROOMSCAN_BACKEND=stub` for the guaranteed fixture demo, or leave CUDA selected and use `points.ply`: splat failure is non-fatal and the viewer automatically falls back to the point cloud. Never try to install gsplat on the Mac.

## Offline reconstruction

```bash
python reconstruct.py /path/to/images /path/to/output
```

The longest image edge is resized to 518 before VGGT. At most 32 evenly spaced frames are used by default. A CUDA/MPS OOM halves the frame set and retries down to four frames. `confidence_threshold` is a 0–1 rejected percentile because VGGT confidence is not normalized; `0.5` keeps the higher-confidence half of finite points.

Every stage logs wall-clock seconds and updates `meta.json`. All operational knobs—including masking, frame caps, upload limits, motion heuristic, and splat training—live in `config.json`.

## Live success test

1. Start the HTTPS server on the box and open `/status?session=demo` on the presentation browser.
2. Put `https://<box-host>:8443/?session=demo` in a QR code.
3. Have five phones scan it. Each person presses **START**, grants camera/motion permission, and walks sideways in a short arc for 15 seconds. A red warning appears when motion looks like rotation without translation.
4. Confirm roughly 20 frames per phone on the status page. Upload completion never starts reconstruction automatically.
5. Press **Reconstruct** once. Watch stage times in stdout. Open the viewer as soon as `viewer_ready` becomes true; `points.ply` is usable even if the optional 3,000-iteration splat is still training or fails.
6. The viewer should show a recognizable occupied room within 90 seconds of the last upload, with people removed, and start on the recovered-camera flythrough. Use free orbit only as a secondary check.

The acceptance run must be performed on CUDA. Stub verification proves the capture/server/viewer contract, not reconstruction quality. MPS verification proves only that imports and a small forward pass can execute.

## Tests

```bash
ROOMSCAN_BACKEND=stub python -m unittest discover -s tests -v
```

The suite checks fixture validity, backend selection without ML dependencies, durable late-stage failure behavior, OOM subsampling, local-only viewer assets, WebSocket upload placement, manual triggering, and the complete stub pipeline.
