# Medical Emergency Gesture System - Viva Guide

This system uses computer vision and hand landmark detection to translate physical gestures into immediate medical requests.

## Gesture-to-Meaning Mapping

| Hand Gesture | Meaning | Use Case |
| :--- | :--- | :--- |
| **Open Palm** | HELP | General assistance required. |
| **Fist** | PAIN | Indicates acute physical discomfort. |
| **1 Finger (Index)** | WATER | Requesting hydration. |
| **2 Fingers (V-Sign)** | DOCTOR | Requesting professional medical attention. |
| **5 Fingers (Shaking)** | EMERGENCY | Critical alert! Triggers on-screen visual warning. |

## Technical Implementation Summary

1.  **Hand Landmark Detection**: Uses MediaPipe to detect 21 hand keypoints.
2.  **ML Classification**: The coordinates are normalized and passed into a Lightweight Multi-Layer Perceptron (MLP).
3.  **Label Mapping**: Class IDs from the model are mapped to medical terminology defined in `keypoint_classifier_label.csv`.
4.  **Alert System**: A special trigger in the code `app.py` detects the "EMERGENCY" string and overlays a high-priority red alert on the display.
