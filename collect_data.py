#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
collect_data.py
───────────────
Interactive data collection tool for missing gesture classes.
Run this to record keypoint data for EMERGENCY, OK, STOP, CALL, MEDICINE, BREATH.

Controls:
  SPACE  - Start/stop recording for current gesture
  N      - Next gesture
  Q      - Quit and save
"""

import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import csv
import copy
import time
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'hand_landmarker.task')
CSV_PATH   = os.path.join(BASE_DIR, 'model', 'keypoint_classifier', 'keypoint.csv')

LABELS = ['HELP', 'PAIN', 'WATER', 'DOCTOR', 'EMERGENCY', 'OK', 'STOP', 'CALL', 'MEDICINE', 'BREATH']

# Only collect for missing classes
MISSING_CLASSES = {
    4: 'EMERGENCY',
    5: 'OK',
    6: 'STOP',
    7: 'CALL',
    8: 'MEDICINE',
    9: 'BREATH',
}
SAMPLES_PER_CLASS = 1500  # aim for ~1500 samples each


def pre_process_landmark(landmark_list):
    tmp = copy.deepcopy(landmark_list)
    bx, by = tmp[0][0], tmp[0][1]
    for i in range(len(tmp)):
        tmp[i][0] -= bx
        tmp[i][1] -= by
    flat = [n for pt in tmp for n in pt]
    mx = max(map(abs, flat)) or 1.0
    return [n / mx for n in flat]


def main():
    # Count existing samples per class
    class_counts = {i: 0 for i in MISSING_CLASSES}
    try:
        with open(CSV_PATH) as f:
            for row in csv.reader(f):
                idx = int(row[0])
                if idx in class_counts:
                    class_counts[idx] += 1
    except FileNotFoundError:
        pass

    missing_list = [idx for idx, name in MISSING_CLASSES.items()
                    if class_counts[idx] < SAMPLES_PER_CLASS]

    if not missing_list:
        print("✅ All classes already have enough data! Run retrain.py to train the model.")
        return

    current_idx = 0
    class_id = missing_list[current_idx]

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
    recording = False
    collected = {idx: class_counts[idx] for idx in missing_list}

    print(f"\n🎯 Starting data collection for missing gestures")
    print(f"   Press SPACE to start/stop recording | N for next | Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        rgb   = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts     = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_img, ts)

        gesture_name = MISSING_CLASSES[class_id]
        needed  = SAMPLES_PER_CLASS - collected[class_id]
        done    = collected[class_id]

        # Draw landmarks + info
        if result.hand_landmarks:
            lm_list = [[lm.x * frame.shape[1], lm.y * frame.shape[0]]
                       for lm in result.hand_landmarks[0]]
            for lm in lm_list:
                cv.circle(frame, (int(lm[0]), int(lm[1])), 5, (0, 255, 0), -1)

            if recording:
                processed = pre_process_landmark([[lm.x, lm.y] for lm in result.hand_landmarks[0]])
                with open(CSV_PATH, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([class_id] + processed)
                collected[class_id] += 1

                if collected[class_id] >= SAMPLES_PER_CLASS:
                    recording = False
                    print(f"  ✅ {gesture_name} complete! ({collected[class_id]} samples)")
                    current_idx += 1
                    if current_idx >= len(missing_list):
                        print("\n🎉 All gestures collected! Run: python retrain.py")
                        break
                    class_id = missing_list[current_idx]

        # HUD overlay
        color = (0, 0, 200) if recording else (50, 50, 50)
        cv.rectangle(frame, (0, 0), (frame.shape[1], 100), color, -1)

        status = "🔴 RECORDING" if recording else "⬜ READY (press SPACE)"
        cv.putText(frame, f"Gesture: {gesture_name} [{class_id}]", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv.putText(frame, f"Collected: {done} / {SAMPLES_PER_CLASS}  {status}", (10, 65),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Progress bar
        progress = int((done / SAMPLES_PER_CLASS) * frame.shape[1])
        cv.rectangle(frame, (0, 95), (progress, 100), (0, 255, 0), -1)

        # Instruction panel
        remaining = [f"{MISSING_CLASSES[i]}({collected[i]}/{SAMPLES_PER_CLASS})"
                     for i in missing_list]
        cv.putText(frame, "Remaining: " + " | ".join(remaining),
                   (10, frame.shape[0] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv.imshow("Data Collector - Sign Language", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord(' '):
            recording = not recording
            print(f"  {'▶ Recording' if recording else '⏸ Paused'} {gesture_name}")
        elif key == ord('n'):
            recording = False
            current_idx = min(current_idx + 1, len(missing_list) - 1)
            class_id = missing_list[current_idx]
            print(f"  ⏭ Switched to: {MISSING_CLASSES[class_id]}")
        elif key == ord('q'):
            print("\n💾 Data saved. Run: python retrain.py")
            break

    cap.release()
    cv.destroyAllWindows()
    landmarker.close()

    print("\n📊 Collection summary:")
    for idx in missing_list:
        name = MISSING_CLASSES[idx]
        cnt  = collected[idx]
        status = "✅" if cnt >= SAMPLES_PER_CLASS else f"⚠️  ({cnt}/{SAMPLES_PER_CLASS})"
        print(f"  [{idx}] {name}: {cnt} samples {status}")


if __name__ == '__main__':
    main()
