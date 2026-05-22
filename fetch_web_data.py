#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fetch_web_data.py
[*]
Downloads pre-extracted MediaPipe hand landmark data from the
HaGRID dataset on Hugging Face (Vincent-luo/hagrid-mediapipe-hands)
and converts it into our keypoint.csv format.

Gesture mapping from HaGRID [*] our labels:
  HaGRID "stop"  [*] class 6 (STOP)
  HaGRID "ok"    [*] class 5 (OK)
  HaGRID "call"  [*] class 7 (CALL)
  HaGRID "palm"  [*] class 4 (EMERGENCY)  [open palm = all fingers spread]
  HaGRID "one"   [*] class 8 (MEDICINE)   [single index finger up]
  HaGRID "fist"  [*] class 9 (BREATH)     [closed fist = exhale/breath hold)
"""

import os
import csv
import copy
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, 'model', 'keypoint_classifier', 'keypoint.csv')
LABELS    = ['HELP','PAIN','WATER','DOCTOR','EMERGENCY','OK','STOP','CALL','MEDICINE','BREATH']

# HaGRID gesture name [*] our class index
HAGRID_MAP = {
    'stop'  : 6,   # STOP
    'ok'    : 5,   # OK
    'call'  : 7,   # CALL
    'palm'  : 4,   # EMERGENCY (open palm, all 5 fingers spread)
    'one'   : 8,   # MEDICINE  (index finger up)
    'fist'  : 9,   # BREATH    (closed fist)
}

SAMPLES_PER_CLASS = 1500


def normalize_landmarks(raw_lm):
    """
    raw_lm: list of 21 dicts with keys 'x','y' (or list of [x,y] pairs)
    Returns 42 normalized floats, same as server.py preprocessing.
    """
    if isinstance(raw_lm[0], dict):
        pts = [[p['x'], p['y']] for p in raw_lm]
    else:
        pts = [[p[0], p[1]] for p in raw_lm]

    tmp = copy.deepcopy(pts)
    bx, by = tmp[0][0], tmp[0][1]
    for i in range(len(tmp)):
        tmp[i][0] -= bx
        tmp[i][1] -= by
    flat = [n for pt in tmp for n in pt]
    mx = max(map(abs, flat)) or 1.0
    return [n / mx for n in flat]


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


def install_deps():
    import subprocess, sys
    pkgs = ['datasets', 'huggingface_hub', 'pyarrow', 'pandas']
    print("[*] Installing required packages...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs)
    print("[OK] Packages ready\n")


def main():
    install_deps()

    from datasets import load_dataset
    import pandas as pd

    existing = count_existing()
    print("[*] Current data counts:")
    for idx, name in enumerate(LABELS):
        print(f"  [{idx}] {name}: {existing[idx]} samples")

    needed = {idx: SAMPLES_PER_CLASS - existing[idx]
              for idx in HAGRID_MAP.values()
              if existing[idx] < SAMPLES_PER_CLASS}

    if not needed:
        print("\n[*] All classes already have enough data! Run: python retrain.py")
        return

    print(f"\n[*] Downloading HaGRID MediaPipe landmarks from Hugging Face...")
    print("   Dataset: Vincent-luo/hagrid-mediapipe-hands")
    print("   This may take a few minutes on first run (cached after that)\n")

    # Load only the gestures we need
    hagrid_gestures_needed = [g for g, cls in HAGRID_MAP.items()
                               if cls in needed]
    print(f"   Fetching gestures: {hagrid_gestures_needed}\n")

    added = {idx: 0 for idx in needed}

    with open(CSV_PATH, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for gesture_name in hagrid_gestures_needed:
            class_id = HAGRID_MAP[gesture_name]
            label_name = LABELS[class_id]
            n_needed = needed.get(class_id, 0)

            if n_needed <= 0:
                continue

            print(f"[*] Fetching '{gesture_name}' [*] [{class_id}] {label_name} ({n_needed} needed)...")

            try:
                # Load this specific gesture split
                ds = load_dataset(
                    "Vincent-luo/hagrid-mediapipe-hands",
                    split=f"train",
                    trust_remote_code=True,
                )

                # Filter to only this gesture
                ds_filtered = ds.filter(lambda x: x['label'] == gesture_name)
                count = 0

                for sample in ds_filtered:
                    if count >= n_needed:
                        break

                    # landmarks field: list of 21 {x, y, z} dicts
                    landmarks = sample.get('landmarks') or sample.get('hand_landmarks')
                    if not landmarks or len(landmarks) < 21:
                        continue

                    try:
                        normalized = normalize_landmarks(landmarks[:21])
                        writer.writerow([class_id] + normalized)
                        count += 1
                        added[class_id] = added.get(class_id, 0) + 1
                    except Exception:
                        continue

                print(f"  [*] Added {count} samples for {label_name}")

            except Exception as e:
                print(f"  [*]  Error fetching '{gesture_name}': {e}")
                print(f"     Trying alternative approach...")
                _try_alternative(gesture_name, class_id, n_needed, writer, added)

    print("\n[*] Final summary:")
    final = count_existing()
    for idx, name in enumerate(LABELS):
        status = "[*]" if final[idx] >= SAMPLES_PER_CLASS else f"[*]  {final[idx]}/{SAMPLES_PER_CLASS}"
        print(f"  [{idx}] {name}: {final[idx]} samples  {status}")

    missing_still = [LABELS[i] for i in range(len(LABELS)) if final[i] < SAMPLES_PER_CLASS]
    if missing_still:
        print(f"\n[*]  Still missing data for: {missing_still}")
        print("   Run collect_data.py to record these manually.")
    else:
        print("\n[*] All classes have enough data!")
        print("   Now run: python retrain.py")


def _try_alternative(gesture_name, class_id, n_needed, writer, added):
    """
    Alternative: generate geometrically realistic synthetic landmarks
    based on known hand poses for each gesture.
    This ensures the model always has training data even if download fails.
    """
    import random

    label_name = LABELS[class_id]
    print(f"  [*] Generating synthetic landmarks for '{label_name}'...")

    # Define canonical normalized landmark positions for each gesture
    # Format: 21 [x, y] pairs, wrist at origin after normalization
    CANONICAL_POSES = {
        # STOP (class 6): flat open palm facing camera
        6: lambda: _perturb([
            [0,0],[-.1,.3],[-.15,.6],[-.15,.85],[-.15,1.0],
            [.05,.55],[.05,.85],[.05,1.0],[.05,1.1],
            [.2,.55],[.2,.85],[.2,1.0],[.2,1.1],
            [.35,.5],[.35,.8],[.35,.95],[.35,1.05],
            [.45,.4],[.45,.65],[.45,.8],[.45,.9]
        ]),
        # OK (class 5): thumb + index touching, others extended
        5: lambda: _perturb([
            [0,0],[-.05,.3],[-.1,.55],[-.1,.7],[-.08,.82],
            [.1,.5],[.05,.75],[.0,.9],[-.05,.98],
            [.2,.5],[.2,.82],[.2,.98],[.2,1.05],
            [.32,.45],[.32,.75],[.32,.9],[.32,.98],
            [.4,.35],[.42,.6],[.42,.75],[.42,.85]
        ]),
        # CALL (class 7): thumb + pinky out, others folded
        7: lambda: _perturb([
            [0,0],[-.1,.3],[-.18,.55],[-.2,.65],[-.18,.5],
            [.05,.45],[.05,.55],[.05,.5],[.05,.45],
            [.15,.45],[.15,.52],[.15,.48],[.15,.43],
            [.27,.42],[.27,.5],[.27,.46],[.27,.42],
            [.38,.35],[.45,.55],[.45,.72],[.45,.88]
        ]),
        # EMERGENCY (class 4): all 5 fingers fully spread
        4: lambda: _perturb([
            [0,0],[-.18,.25],[-.28,.5],[-.3,.75],[-.28,.95],
            [-.05,.55],[-.08,.88],[-.08,1.05],[-.06,1.18],
            [.1,.6],[.08,.95],[.08,1.12],[.08,1.22],
            [.28,.52],[.28,.85],[.28,1.0],[.28,1.1],
            [.42,.35],[.48,.6],[.48,.78],[.48,.9]
        ]),
        # MEDICINE (class 8): only index finger up
        8: lambda: _perturb([
            [0,0],[-.08,.28],[-.15,.5],[-.17,.62],[-.15,.48],
            [.05,.5],[.05,.82],[.05,1.0],[.05,1.12],
            [.17,.48],[.17,.58],[.17,.52],[.17,.47],
            [.3,.45],[.3,.55],[.3,.5],[.3,.45],
            [.4,.32],[.42,.5],[.42,.62],[.42,.72]
        ]),
        # BREATH (class 9): closed relaxed fist
        9: lambda: _perturb([
            [0,0],[-.05,.28],[-.1,.48],[-.12,.55],[-.1,.45],
            [.05,.4],[.06,.55],[.05,.5],[.04,.42],
            [.15,.4],[.16,.55],[.15,.5],[.14,.42],
            [.26,.38],[.27,.52],[.26,.47],[.25,.4],
            [.35,.28],[.38,.42],[.38,.5],[.37,.42]
        ]),
    }

    def _perturb(pts, noise=0.04):
        return [[p[0] + random.uniform(-noise, noise),
                 p[1] + random.uniform(-noise, noise)] for p in pts]

    pose_fn = CANONICAL_POSES.get(class_id)
    if not pose_fn:
        print(f"  [*] No canonical pose for class {class_id}")
        return

    count = 0
    for _ in range(n_needed):
        pts = pose_fn()
        normalized = normalize_landmarks(pts)
        writer.writerow([class_id] + normalized)
        count += 1
        added[class_id] = added.get(class_id, 0) + 1

    print(f"  [*] Generated {count} synthetic samples for {label_name}")


if __name__ == '__main__':
    main()
