#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
collect_real.py
---------------
Records REAL hand gesture data from your webcam for the 5 gestures
that only have synthetic data. Replaces synthetic data with real recordings.

Controls:
  SPACE  - Start/stop recording
  N      - Next gesture
  Q      - Quit and save
"""
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import csv, copy, time, os, sys

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'hand_landmarker.task')
CSV_PATH   = os.path.join(BASE_DIR, 'model', 'keypoint_classifier', 'keypoint.csv')

ALL_LABELS = ['HELP','PAIN','WATER','DOCTOR','EMERGENCY','OK','STOP','CALL','MEDICINE','BREATH']

# Only record these 5 (synthetic ones that don't match real hands)
RECORD_CLASSES = {
    4: 'EMERGENCY  [all 5 fingers spread wide]',
    5: 'OK         [thumb + index tip touching, others up]',
    6: 'STOP       [flat open palm facing camera]',
    7: 'CALL       [thumb + pinky extended, others folded]',
    8: 'MEDICINE   [only index finger pointing up]',
    9: 'BREATH     [relaxed slightly open hand]',
}
SAMPLES_PER_CLASS = 500  # 500 real samples easily beats 1500 synthetic


def pre_process(landmark_list):
    tmp = copy.deepcopy(landmark_list)
    bx, by = tmp[0][0], tmp[0][1]
    for i in range(len(tmp)):
        tmp[i][0] -= bx
        tmp[i][1] -= by
    flat = [n for pt in tmp for n in pt]
    mx = max(map(abs, flat)) or 1.0
    return [n / mx for n in flat]


def remove_synthetic_and_load_real(target_classes):
    """Remove old synthetic rows for target classes, keep everything else."""
    print(f"[*] Removing old synthetic data for classes: {target_classes}")
    kept_rows = []
    removed = {c: 0 for c in target_classes}
    real_kept = {c: 0 for c in target_classes}

    try:
        with open(CSV_PATH, encoding='utf-8-sig') as f:
            for row in csv.reader(f):
                if not row:
                    continue
                idx = int(row[0])
                if idx in target_classes:
                    # We'll drop ALL existing data for target classes
                    # (it was all synthetic anyway)
                    removed[idx] = removed.get(idx, 0) + 1
                else:
                    kept_rows.append(row)
    except FileNotFoundError:
        pass

    # Rewrite CSV without synthetic data for target classes
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in kept_rows:
            writer.writerow(row)

    for c, n in removed.items():
        print(f"  Removed {n} synthetic samples for [{c}] {ALL_LABELS[c]}")
    return removed


def main():
    class_list = sorted(RECORD_CLASSES.keys())

    print("=" * 60)
    print("REAL DATA COLLECTOR - Sign Language Recognition")
    print("=" * 60)
    print(f"\nTarget: {SAMPLES_PER_CLASS} real samples per gesture")
    print("\nGestures to record:")
    for idx, desc in RECORD_CLASSES.items():
        print(f"  [{idx}] {desc}")
    print("\nThis will REPLACE the synthetic data with your real gestures.")
    answer = input("\nProceed? (y/n): ").strip().lower()
    if answer != 'y':
        print("Cancelled.")
        return

    # Remove old synthetic data for target classes
    remove_synthetic_and_load_real(class_list)
    print()

    # Setup MediaPipe
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv.VideoCapture(0)
    current_idx = 0
    class_id = class_list[current_idx]
    collected = {idx: 0 for idx in class_list}
    recording = False

    print("\nCAMERA WINDOW OPEN - Controls:")
    print("  SPACE = start/stop recording")
    print("  N     = next gesture")
    print("  Q     = quit and retrain\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        rgb   = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts    = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_img, ts)

        h, w = frame.shape[:2]
        gesture_name = ALL_LABELS[class_id]
        done   = collected[class_id]
        needed = SAMPLES_PER_CLASS

        # Detect hand
        hand_detected = bool(result.hand_landmarks)

        if hand_detected:
            lm_list = [[lm.x * w, lm.y * h] for lm in result.hand_landmarks[0]]
            for lm in lm_list:
                cv.circle(frame, (int(lm[0]), int(lm[1])), 4, (0, 255, 0), -1)

            if recording:
                processed = pre_process([[lm.x, lm.y] for lm in result.hand_landmarks[0]])
                with open(CSV_PATH, 'a', newline='') as f:
                    csv.writer(f).writerow([class_id] + processed)
                collected[class_id] += 1
                done += 1

                if collected[class_id] >= needed:
                    recording = False
                    print(f"  [DONE] {gesture_name}: {collected[class_id]} samples recorded!")
                    current_idx += 1
                    if current_idx >= len(class_list):
                        print("\n[*] All gestures recorded!")
                        break
                    class_id = class_list[current_idx]
                    gesture_name = ALL_LABELS[class_id]
                    print(f"  Next: [{class_id}] {RECORD_CLASSES[class_id]}")

        # ── HUD ──────────────────────────────────────────────────────────────
        # Header bar
        bar_color = (0, 0, 180) if recording else (40, 40, 40)
        cv.rectangle(frame, (0, 0), (w, 110), bar_color, -1)

        cv.putText(frame, f"Gesture [{class_id}]: {gesture_name}",
                   (10, 32), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        hand_status = "HAND DETECTED" if hand_detected else "NO HAND - Show your hand!"
        hand_color  = (0, 255, 100) if hand_detected else (0, 100, 255)
        cv.putText(frame, hand_status, (10, 62),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65, hand_color, 2)

        rec_text = "RECORDING..." if recording else "PRESS SPACE to record"
        rec_color = (0, 255, 255) if recording else (200, 200, 200)
        cv.putText(frame, rec_text, (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65, rec_color, 2)

        # Progress bar
        prog_w = int((done / needed) * w)
        cv.rectangle(frame, (0, 108), (prog_w, 114), (0, 200, 0), -1)
        cv.putText(frame, f"{done}/{needed}", (w//2 - 40, 107),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Bottom status
        remaining = " | ".join(
            f"{ALL_LABELS[i]}:{collected[i]}"
            for i in class_list
        )
        cv.rectangle(frame, (0, h-28), (w, h), (20,20,20), -1)
        cv.putText(frame, remaining, (8, h-8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)

        # Gesture hint
        hint_lines = RECORD_CLASSES[class_id].split('[')
        if len(hint_lines) > 1:
            hint = hint_lines[1].rstrip(']')
            cv.putText(frame, f"HOW: {hint}", (10, h-38),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,255), 1)

        cv.imshow("REAL Data Collector", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord(' '):
            if not hand_detected:
                print("  [!] No hand detected - show your hand first!")
            else:
                recording = not recording
                print(f"  {'RECORDING' if recording else 'PAUSED'} [{gesture_name}]")
        elif key == ord('n'):
            recording = False
            current_idx = min(current_idx + 1, len(class_list) - 1)
            class_id = class_list[current_idx]
            print(f"  Skipped to: [{class_id}] {RECORD_CLASSES[class_id]}")
        elif key == ord('q'):
            print("\n[*] Quit by user.")
            break

    cap.release()
    cv.destroyAllWindows()
    landmarker.close()

    # Summary
    print("\n--- Collection Summary ---")
    for idx in class_list:
        n = collected[idx]
        status = "OK" if n >= SAMPLES_PER_CLASS else f"LOW ({n}/{SAMPLES_PER_CLASS})"
        print(f"  [{idx}] {ALL_LABELS[idx]}: {n} samples [{status}]")

    total = sum(collected.values())
    if total > 0:
        print(f"\nTotal recorded: {total} samples")
        print("\nNow run to retrain:")
        print("  .\\venv311\\Scripts\\python.exe retrain.py")
    else:
        print("\n[!] No data recorded. Run again and press SPACE to record.")


if __name__ == '__main__':
    main()
