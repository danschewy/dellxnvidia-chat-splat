#!/usr/bin/env python3
"""Print a loud, actionable ROOMSCAN readiness report."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import time

from backends import select_backend
from models import import_gsplat_checked, load_vggt, load_yolo, vggt_path, yolo_path
from roomscan_io import load_config, resolve_weights_dir


ROOT = Path(__file__).resolve().parent


def status(label: str, state: str, detail: str) -> None:
    print(f"{label:<22} {state:<8} {detail}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--smoke-models", action="store_true", help="load models and run a four-image VGGT forward pass")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    weights = resolve_weights_dir(config, ROOT)
    print("\nROOMSCAN ENVIRONMENT CHECK")
    print("=" * 70)
    status("platform", "INFO", f"{platform.system()} {platform.machine()} / Python {platform.python_version()}")
    backend = select_backend(config, ROOT)
    status("selected backend", "OK", backend.name)

    try:
        import torch
    except ImportError:
        status("torch", "MISSING", "install requirements-mac.txt or requirements-cuda.txt")
        torch = None
    else:
        status("torch", "OK", torch.__version__)
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            status("accelerator", "OK", f"{properties.name}, {properties.total_memory / 2**30:.1f} GiB")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            status("accelerator", "OK", "Apple MPS (unified memory; capacity not exposed by torch)")
        else:
            status("accelerator", "NONE", "CPU only; stub backend is expected")

    try:
        status("VGGT weights", "OK", str(vggt_path(weights)))
    except Exception as exc:
        status("VGGT weights", "MISSING", str(exc))
    try:
        status("YOLO weights", "OK", str(yolo_path(weights)))
    except Exception as exc:
        status("YOLO weights", "MISSING", str(exc))

    try:
        import_gsplat_checked()
    except Exception as exc:
        state = "EXPECTED" if backend.name != "cuda" else "FAIL"
        status("gsplat extension", state, str(exc))
        if backend.name == "cuda":
            print("!!! CUDA WAS DETECTED BUT GSPLAT IS NOT BUILT. SPLAT TRAINING WILL NOT WORK. !!!")
    else:
        status("gsplat extension", "OK", "imported rasterization entry point")

    if arguments.smoke_models and backend.name in {"cuda", "mps"}:
        assert torch is not None
        started = time.perf_counter()
        try:
            model = load_vggt(weights, backend.name)
            dummy = torch.zeros(
                (1, 4, 3, int(config["vggt_resolution"]), int(config["vggt_resolution"])),
                device=backend.name,
            )
            dtype = torch.bfloat16 if backend.name == "cuda" else torch.float16
            with torch.inference_mode(), torch.autocast(device_type=backend.name, dtype=dtype):
                output = model(dummy)
            required = {"pose_enc", "world_points", "world_points_conf"}
            missing = required - output.keys()
            if missing:
                raise RuntimeError(f"forward output missing keys: {sorted(missing)}")
        except Exception as exc:
            status("VGGT 4-frame smoke", "FAIL", str(exc))
        else:
            status("VGGT 4-frame smoke", "OK", f"{time.perf_counter() - started:.2f}s")
        try:
            load_yolo(weights)
        except Exception as exc:
            status("YOLO load", "FAIL", str(exc))
        else:
            status("YOLO load", "OK", "local yolo11s-seg loaded")
    else:
        status("model smoke tests", "SKIPPED", "pass --smoke-models on CUDA/MPS")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
