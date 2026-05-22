#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_synthetic.py
---------------------
Generates realistic synthetic hand landmark data for the 6 missing
gesture classes and appends them to keypoint.csv, then retrains the model.

Uses geometrically accurate MediaPipe-style 21-point hand landmarks
with random perturbations to simulate real variation.

Run:  python generate_synthetic.py
Then: python retrain.py
"""
import os, csv, copy, random
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'model', 'keypoint_classifier', 'keypoint.csv')
LABELS   = ['HELP','PAIN','WATER','DOCTOR','EMERGENCY','OK','STOP','CALL','MEDICINE','BREATH']

SAMPLES = 1500   # samples per missing class

# ── Canonical hand poses (21 landmarks, wrist at [0,0]) ──────────────────────
# Each landmark: [x, y] roughly in [-1..1] space before normalization
# Based on real MediaPipe hand landmark topology

POSES = {
    # EMERGENCY [4]: All 5 fingers spread wide open (starfish hand)
    4: [
        [ 0.00,  0.00],  # 0  wrist
        [-0.18,  0.25],  # 1  thumb CMC
        [-0.30,  0.50],  # 2  thumb MCP
        [-0.38,  0.72],  # 3  thumb IP
        [-0.42,  0.92],  # 4  thumb tip
        [-0.08,  0.60],  # 5  index MCP
        [-0.10,  0.90],  # 6  index PIP
        [-0.10,  1.12],  # 7  index DIP
        [-0.10,  1.28],  # 8  index tip
        [ 0.08,  0.62],  # 9  middle MCP
        [ 0.08,  0.94],  # 10 middle PIP
        [ 0.08,  1.18],  # 11 middle DIP
        [ 0.08,  1.34],  # 12 middle tip
        [ 0.24,  0.58],  # 13 ring MCP
        [ 0.26,  0.88],  # 14 ring PIP
        [ 0.26,  1.10],  # 15 ring DIP
        [ 0.26,  1.24],  # 16 ring tip
        [ 0.38,  0.45],  # 17 pinky MCP
        [ 0.44,  0.68],  # 18 pinky PIP
        [ 0.44,  0.85],  # 19 pinky DIP
        [ 0.44,  0.98],  # 20 pinky tip
    ],

    # OK [5]: Thumb + index forming a circle, other fingers extended
    5: [
        [ 0.00,  0.00],  # 0  wrist
        [-0.12,  0.28],  # 1  thumb CMC
        [-0.18,  0.50],  # 2  thumb MCP
        [-0.16,  0.68],  # 3  thumb IP
        [-0.10,  0.80],  # 4  thumb tip (touching index tip)
        [ 0.05,  0.52],  # 5  index MCP
        [ 0.02,  0.72],  # 6  index PIP
        [-0.02,  0.82],  # 7  index DIP
        [-0.08,  0.80],  # 8  index tip (touching thumb)
        [ 0.18,  0.54],  # 9  middle MCP
        [ 0.18,  0.86],  # 10 middle PIP
        [ 0.18,  1.05],  # 11 middle DIP
        [ 0.18,  1.18],  # 12 middle tip
        [ 0.30,  0.50],  # 13 ring MCP
        [ 0.30,  0.82],  # 14 ring PIP
        [ 0.30,  1.00],  # 15 ring DIP
        [ 0.30,  1.12],  # 16 ring tip
        [ 0.40,  0.38],  # 17 pinky MCP
        [ 0.44,  0.62],  # 18 pinky PIP
        [ 0.44,  0.78],  # 19 pinky DIP
        [ 0.44,  0.90],  # 20 pinky tip
    ],

    # STOP [6]: Flat open palm facing camera, fingers together
    6: [
        [ 0.00,  0.00],  # 0  wrist
        [-0.10,  0.28],  # 1  thumb CMC
        [-0.18,  0.52],  # 2  thumb MCP
        [-0.22,  0.72],  # 3  thumb IP
        [-0.24,  0.90],  # 4  thumb tip
        [-0.04,  0.58],  # 5  index MCP
        [-0.04,  0.88],  # 6  index PIP
        [-0.04,  1.08],  # 7  index DIP
        [-0.04,  1.22],  # 8  index tip
        [ 0.08,  0.60],  # 9  middle MCP
        [ 0.08,  0.92],  # 10 middle PIP
        [ 0.08,  1.12],  # 11 middle DIP
        [ 0.08,  1.28],  # 12 middle tip
        [ 0.20,  0.58],  # 13 ring MCP
        [ 0.20,  0.88],  # 14 ring PIP
        [ 0.20,  1.06],  # 15 ring DIP
        [ 0.20,  1.20],  # 16 ring tip
        [ 0.30,  0.48],  # 17 pinky MCP
        [ 0.30,  0.72],  # 18 pinky PIP
        [ 0.30,  0.88],  # 19 pinky DIP
        [ 0.30,  1.00],  # 20 pinky tip
    ],

    # CALL [7]: Thumb + pinky extended (phone gesture), others folded
    7: [
        [ 0.00,  0.00],  # 0  wrist
        [-0.12,  0.28],  # 1  thumb CMC
        [-0.22,  0.52],  # 2  thumb MCP
        [-0.30,  0.72],  # 3  thumb IP
        [-0.36,  0.90],  # 4  thumb tip (extended)
        [ 0.05,  0.50],  # 5  index MCP
        [ 0.04,  0.62],  # 6  index PIP (folded)
        [ 0.04,  0.55],  # 7  index DIP
        [ 0.04,  0.48],  # 8  index tip
        [ 0.16,  0.52],  # 9  middle MCP
        [ 0.15,  0.64],  # 10 middle PIP (folded)
        [ 0.15,  0.57],  # 11 middle DIP
        [ 0.15,  0.50],  # 12 middle tip
        [ 0.26,  0.50],  # 13 ring MCP
        [ 0.25,  0.62],  # 14 ring PIP (folded)
        [ 0.25,  0.56],  # 15 ring DIP
        [ 0.25,  0.50],  # 16 ring tip
        [ 0.35,  0.40],  # 17 pinky MCP
        [ 0.40,  0.62],  # 18 pinky PIP (extended)
        [ 0.44,  0.80],  # 19 pinky DIP
        [ 0.46,  0.96],  # 20 pinky tip (extended)
    ],

    # MEDICINE [8]: Only index finger pointing up
    8: [
        [ 0.00,  0.00],  # 0  wrist
        [-0.08,  0.28],  # 1  thumb CMC
        [-0.16,  0.50],  # 2  thumb MCP
        [-0.18,  0.62],  # 3  thumb IP (slightly folded)
        [-0.16,  0.52],  # 4  thumb tip
        [ 0.05,  0.52],  # 5  index MCP
        [ 0.05,  0.82],  # 6  index PIP (extended)
        [ 0.05,  1.05],  # 7  index DIP
        [ 0.05,  1.20],  # 8  index tip (pointing up)
        [ 0.16,  0.50],  # 9  middle MCP
        [ 0.15,  0.62],  # 10 middle PIP (folded)
        [ 0.15,  0.55],  # 11 middle DIP
        [ 0.15,  0.48],  # 12 middle tip
        [ 0.26,  0.48],  # 13 ring MCP
        [ 0.25,  0.60],  # 14 ring PIP (folded)
        [ 0.25,  0.53],  # 15 ring DIP
        [ 0.25,  0.47],  # 16 ring tip
        [ 0.34,  0.38],  # 17 pinky MCP
        [ 0.34,  0.50],  # 18 pinky PIP (folded)
        [ 0.34,  0.44],  # 19 pinky DIP
        [ 0.34,  0.38],  # 20 pinky tip
    ],

    # BREATH [9]: Relaxed open hand, fingers slightly curved (natural rest)
    9: [
        [ 0.00,  0.00],  # 0  wrist
        [-0.10,  0.26],  # 1  thumb CMC
        [-0.20,  0.48],  # 2  thumb MCP
        [-0.26,  0.65],  # 3  thumb IP
        [-0.28,  0.80],  # 4  thumb tip
        [-0.02,  0.55],  # 5  index MCP
        [-0.05,  0.80],  # 6  index PIP
        [-0.06,  0.98],  # 7  index DIP
        [-0.06,  1.10],  # 8  index tip
        [ 0.10,  0.58],  # 9  middle MCP
        [ 0.08,  0.85],  # 10 middle PIP
        [ 0.07,  1.05],  # 11 middle DIP
        [ 0.07,  1.18],  # 12 middle tip
        [ 0.22,  0.55],  # 13 ring MCP
        [ 0.22,  0.80],  # 14 ring PIP
        [ 0.22,  0.98],  # 15 ring DIP
        [ 0.22,  1.10],  # 16 ring tip
        [ 0.32,  0.44],  # 17 pinky MCP
        [ 0.35,  0.65],  # 18 pinky PIP
        [ 0.36,  0.80],  # 19 pinky DIP
        [ 0.36,  0.92],  # 20 pinky tip
    ],
}


def normalize(pts):
    tmp = copy.deepcopy(pts)
    bx, by = tmp[0][0], tmp[0][1]
    for i in range(len(tmp)):
        tmp[i][0] -= bx
        tmp[i][1] -= by
    flat = [n for pt in tmp for n in pt]
    mx = max(map(abs, flat)) or 1.0
    return [n / mx for n in flat]


def perturb(pts, noise=0.05, finger_scale_noise=0.12):
    """Add realistic variation: global noise + per-finger length variation"""
    result = []
    # Global hand rotation variation (small angle)
    angle = random.uniform(-0.25, 0.25)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    for i, p in enumerate(pts):
        # Global scale variation
        scale = random.uniform(0.85, 1.15)
        x = p[0] * scale
        y = p[1] * scale

        # Rotation around wrist
        x_r = x * cos_a - y * sin_a
        y_r = x * sin_a + y * cos_a

        # Per-landmark gaussian noise
        x_r += random.gauss(0, noise)
        y_r += random.gauss(0, noise)

        result.append([x_r, y_r])
    return result


def count_existing():
    counts = {i: 0 for i in range(len(LABELS))}
    try:
        with open(CSV_PATH, encoding='utf-8-sig') as f:
            for row in csv.reader(f):
                if row:
                    idx = int(row[0])
                    if idx in counts:
                        counts[idx] += 1
    except FileNotFoundError:
        pass
    return counts


def main():
    existing = count_existing()

    print("Current class distribution:")
    for idx, name in enumerate(LABELS):
        print(f"  [{idx}] {name}: {existing[idx]} samples")

    missing = {idx: SAMPLES - existing[idx]
               for idx in POSES.keys()
               if existing[idx] < SAMPLES}

    if not missing:
        print("\nAll classes already have enough data!")
        print("Run: python retrain.py")
        return

    print(f"\nGenerating synthetic data for {len(missing)} classes...\n")

    total_added = 0
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)

        for class_id, n_needed in missing.items():
            name = LABELS[class_id]
            base_pose = POSES[class_id]
            count = 0

            for _ in range(n_needed):
                pts = perturb(base_pose)
                normalized = normalize(pts)
                writer.writerow([class_id] + normalized)
                count += 1

            total_added += count
            print(f"  [{class_id}] {name}: +{count} samples  DONE")

    print(f"\nTotal added: {total_added} samples")

    # Verify
    final = count_existing()
    print("\nFinal distribution:")
    all_good = True
    for idx, name in enumerate(LABELS):
        status = "OK" if final[idx] >= SAMPLES else f"LOW ({final[idx]}/{SAMPLES})"
        print(f"  [{idx}] {name}: {final[idx]} samples  [{status}]")
        if final[idx] < SAMPLES:
            all_good = False

    if all_good:
        print("\nAll classes ready! Now run: python retrain.py")
    else:
        print("\nSome classes still need manual data collection.")


if __name__ == '__main__':
    main()
