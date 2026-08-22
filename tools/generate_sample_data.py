#!/usr/bin/env python3
"""Generate deterministic, dependency-free ROOMSCAN demo fixtures."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sample_data"


def normalize(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in vector))
    return [value / length for value in vector]


def cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def camera_matrix(position: list[float], target: list[float]) -> list[list[float]]:
    forward = normalize([target[i] - position[i] for i in range(3)])
    right = normalize(cross(forward, [0.0, 1.0, 0.0]))
    down = cross(forward, right)
    return [
        [right[0], down[0], forward[0], position[0]],
        [right[1], down[1], forward[1], position[1]],
        [right[2], down[2], forward[2], position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def add_surface(
    rows: list[tuple[float, float, float, int, int, int]],
    count: int,
    fixed_axis: int,
    fixed_value: float,
    ranges: tuple[tuple[float, float], tuple[float, float]],
    color: tuple[int, int, int],
    rng: random.Random,
) -> None:
    free_axes = [axis for axis in range(3) if axis != fixed_axis]
    for _ in range(count):
        point = [0.0, 0.0, 0.0]
        point[fixed_axis] = fixed_value + rng.uniform(-0.006, 0.006)
        point[free_axes[0]] = rng.uniform(*ranges[0])
        point[free_axes[1]] = rng.uniform(*ranges[1])
        rgb = tuple(max(0, min(255, channel + rng.randint(-8, 8))) for channel in color)
        rows.append((*point, *rgb))


def add_box(
    rows: list[tuple[float, float, float, int, int, int]],
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[int, int, int],
    rng: random.Random,
) -> None:
    mins = [center[i] - size[i] / 2 for i in range(3)]
    maxs = [center[i] + size[i] / 2 for i in range(3)]
    for axis in range(3):
        others = [candidate for candidate in range(3) if candidate != axis]
        ranges = ((mins[others[0]], maxs[others[0]]), (mins[others[1]], maxs[others[1]]))
        add_surface(rows, 450, axis, mins[axis], ranges, color, rng)
        add_surface(rows, 450, axis, maxs[axis], ranges, color, rng)


def generate_points() -> None:
    rng = random.Random(20250822)
    rows: list[tuple[float, float, float, int, int, int]] = []
    add_surface(rows, 5000, 1, 0.0, ((-4.0, 4.0), (-3.0, 3.0)), (96, 82, 70), rng)
    add_surface(rows, 2200, 1, 3.0, ((-4.0, 4.0), (-3.0, 3.0)), (72, 78, 88), rng)
    add_surface(rows, 3000, 0, -4.0, ((0.0, 3.0), (-3.0, 3.0)), (112, 126, 137), rng)
    add_surface(rows, 3000, 0, 4.0, ((0.0, 3.0), (-3.0, 3.0)), (112, 126, 137), rng)
    add_surface(rows, 4000, 2, -3.0, ((-4.0, 4.0), (0.0, 3.0)), (128, 139, 145), rng)
    add_surface(rows, 4000, 2, 3.0, ((-4.0, 4.0), (0.0, 3.0)), (116, 130, 142), rng)
    add_box(rows, (0.6, 0.45, 0.1), (2.8, 0.9, 1.5), (162, 94, 54), rng)
    add_box(rows, (-2.65, 0.75, -1.55), (1.45, 1.5, 0.8), (44, 103, 142), rng)
    add_box(rows, (2.65, 0.35, -1.65), (0.9, 0.7, 0.9), (63, 132, 104), rng)
    with (OUT / "points.ply").open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\ncomment ROOMSCAN synthetic furnished room\n")
        handle.write(f"element vertex {len(rows)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for row in rows:
            handle.write(" ".join(str(round(value, 5)) for value in row) + "\n")


def generate_cameras() -> None:
    cameras = []
    for index in range(25):
        angle = -0.9 + 1.8 * index / 24
        position = [2.8 * math.sin(angle), 1.5 + 0.12 * math.sin(index * 0.7), 2.15 * math.cos(angle)]
        cameras.append(
            {
                "frame": f"frame_{index + 1:03d}.jpg",
                "T_wc": camera_matrix(position, [0.0, 1.05, 0.0]),
                "K": [[465.0, 0.0, 259.0], [0.0, 465.0, 259.0], [0.0, 0.0, 1.0]],
                "confidence": round(0.91 + 0.07 * math.sin((index + 1) / 25 * math.pi), 4),
            }
        )
    with (OUT / "cameras.json").open("w", encoding="utf-8") as handle:
        json.dump(cameras, handle, indent=2)
        handle.write("\n")


def generate_images() -> None:
    frames = OUT / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    width, height = 480, 270
    for index in range(25):
        ppm = frames / f"frame_{index + 1:03d}.ppm"
        jpg = frames / f"frame_{index + 1:03d}.jpg"
        pixels = bytearray()
        shift = int(22 * math.sin(index * 0.24))
        for y in range(height):
            for x in range(width):
                if y < 165:
                    base = (44 + y // 8, 54 + y // 7, 67 + y // 6)
                else:
                    base = (78 + (y - 165) // 3, 67 + (y - 165) // 4, 58 + (y - 165) // 5)
                # Orange table, blue chair, green cabinet: stable visual anchors.
                if 135 + shift < x < 335 + shift and 145 < y < 205:
                    base = (170, 95, 48)
                if 30 + shift // 2 < x < 115 + shift // 2 and 105 < y < 225:
                    base = (42, 101, 145)
                if 375 + shift // 3 < x < 448 + shift // 3 and 118 < y < 226:
                    base = (56, 132, 103)
                vignette = int(22 * (((x - width / 2) / width) ** 2 + ((y - height / 2) / height) ** 2))
                pixels.extend(max(0, channel - vignette) for channel in base)
        with ppm.open("wb") as handle:
            handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            handle.write(pixels)
        subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "70", str(ppm), "--out", str(jpg)], check=True, capture_output=True)
        ppm.unlink()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generate_points()
    generate_cameras()
    generate_images()
    print(f"Generated ROOMSCAN fixtures under {OUT}")


if __name__ == "__main__":
    main()
