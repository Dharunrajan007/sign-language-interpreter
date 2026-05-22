import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import csv
import copy
import time
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from threading import Thread, Lock
import uvicorn
import os


# ─────────────────────────────────────────────────────────────────────────────
# Custom Sign Store — landmark-based nearest-neighbour matching
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_SIGNS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'custom_signs.json')

class CustomSignStore:
    """Stores hand landmark vectors for custom signs and matches via cosine similarity."""
    def __init__(self):
        self._lock = Lock()
        self.signs = {}   # {name: {'icon': str, 'samples': [[float,...]]}}
        self._load()

    def _load(self):
        try:
            if os.path.exists(CUSTOM_SIGNS_FILE):
                with open(CUSTOM_SIGNS_FILE, 'r') as f:
                    self.signs = json.load(f)
                # Migrate old entries that lack 'version' field
                for name, data in self.signs.items():
                    if 'version' not in data:
                        data['version'] = 'hand_only'
        except Exception:
            self.signs = {}

    def _save(self):
        try:
            with open(CUSTOM_SIGNS_FILE, 'w') as f:
                json.dump(self.signs, f)
        except Exception:
            pass

    def add_sample(self, name: str, icon: str, landmarks: list):
        with self._lock:
            if name not in self.signs:
                self.signs[name] = {'icon': icon, 'samples': []}
            self.signs[name]['samples'].append(landmarks)
            if len(self.signs[name]['samples']) > 10:
                self.signs[name]['samples'] = self.signs[name]['samples'][-10:]
            self._save()

    def sample_count(self, name: str) -> int:
        with self._lock:
            return len(self.signs.get(name, {}).get('samples', []))


    def _cosine_sim(self, a, b):
        """Cosine similarity between two vectors of the same length."""
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        na = np.linalg.norm(a_arr)
        nb = np.linalg.norm(b_arr)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a_arr / na, b_arr / nb))

    def match(self, landmarks: list, threshold: float = 0.93) -> str:
        """Match incoming landmarks against all saved templates."""
        with self._lock:
            if not self.signs or not landmarks:
                return ""
            
            best_name, best_score = "", threshold

            for name, data in self.signs.items():
                for sample in data.get('samples', []):
                    try:
                        sim = self._cosine_sim(landmarks, sample)
                        if sim > best_score:
                            best_score, best_name = sim, name
                    except Exception:
                        continue

            return best_name

    def list_signs(self) -> dict:
        with self._lock:
            return {k: {
                'icon': v.get('icon', '✋'),
                'samples': len(v.get('samples', [])),
            } for k, v in self.signs.items()}

    def delete_sign(self, name: str):
        with self._lock:
            self.signs.pop(name, None)
            self._save()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, 'model', 'hand_landmarker.task')
TFLITE_MODEL    = os.path.join(BASE_DIR, 'model', 'keypoint_classifier', 'keypoint_classifier.tflite')
LABEL_PATH      = os.path.join(BASE_DIR, 'model', 'keypoint_classifier', 'keypoint_classifier_label.csv')

app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ─────────────────────────────────────────────────────────────────────────────
# Medical Gesture System (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────
# ── Inference settings ────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.75   # min confidence to accept a gesture
SMOOTHING_WINDOW     = 10     # frames to vote over before confirming

class MedicalSystem:
    def __init__(self):
        self.lock           = Lock()
        self.latest_frame   = None
        self.latest_gesture = ""
        self.fps            = 0
        self.running        = True
        self._vote_buffer   = []   # temporal smoothing buffer
        self.latest_landmarks = []   # current preprocessed landmarks for custom sign capture

        # Load labels
        with open(LABEL_PATH, encoding='utf-8-sig') as f:
            self.labels = [row[0] for row in csv.reader(f)]

        # TFLite inference (use ai_edge_litert if available, fall back to tf.lite)
        try:
            from ai_edge_litert.interpreter import Interpreter
            self.interpreter = Interpreter(model_path=TFLITE_MODEL)
        except ImportError:
            import tensorflow as tf
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)

        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # MediaPipe hand landmarker
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

        # Camera — warm up a few frames
        self.cap = cv.VideoCapture(0)
        for _ in range(5):
            self.cap.read()

    # ── Landmark preprocessing ────────────────────────────────────────────────
    def _pre_process(self, landmark_list):
        tmp = copy.deepcopy(landmark_list)
        bx, by = tmp[0][0], tmp[0][1]
        for i in range(len(tmp)):
            tmp[i][0] -= bx
            tmp[i][1] -= by
        flat = [n for pt in tmp for n in pt]
        mx = max(map(abs, flat)) or 1.0
        return [n / mx for n in flat]

    # ── Main capture/inference loop (runs in background thread) ───────────────
    def process_loop(self):
        start_time  = time.time()
        frame_count = 0

        while self.running:
            ret, image = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            image = cv.flip(image, 1)
            rgb   = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts     = int(time.time() * 1000)
            result = self.landmarker.detect_for_video(mp_img, ts)

            current_processed = []
            raw_gesture = ""
            if result.hand_landmarks:
                lm_list = [[lm.x, lm.y] for lm in result.hand_landmarks[0]]
                processed = self._pre_process(lm_list)
                current_processed = processed
                # 1. Check custom signs FIRST
                try:
                    custom = custom_sign_store.match(processed, threshold=0.92)
                    if custom:
                        raw_gesture = custom
                except Exception:
                    pass

                # 2. If no custom match, run built-in model
                if not raw_gesture:
                    self.interpreter.set_tensor(
                        self.input_details[0]['index'],
                        np.array([processed], dtype=np.float32)
                    )
                    self.interpreter.invoke()
                    scores = np.squeeze(
                        self.interpreter.get_tensor(self.output_details[0]['index'])
                    )
                    confidence = float(np.max(scores))
                    idx        = int(np.argmax(scores))

                    # Only accept if above confidence threshold
                    if confidence >= CONFIDENCE_THRESHOLD:
                        raw_gesture = self.labels[idx]

            # --- Temporal smoothing: vote over last N frames ---
            self._vote_buffer.append(raw_gesture)
            if len(self._vote_buffer) > SMOOTHING_WINDOW:
                self._vote_buffer.pop(0)

            # Confirmed gesture = most common in buffer (ignore empty strings)
            non_empty = [g for g in self._vote_buffer if g]
            if non_empty:
                from collections import Counter
                winner, count = Counter(non_empty).most_common(1)[0]
                # Require majority (>50%) to confirm
                gesture_text = winner if count > SMOOTHING_WINDOW // 2 else ""
            else:
                gesture_text = ""

            # FPS counter
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                self.fps       = int(frame_count / elapsed)
                frame_count    = 0
                start_time     = time.time()

            _, buf = cv.imencode('.jpg', image)
            with self.lock:
                self.latest_gesture   = gesture_text
                self.latest_frame     = buf.tobytes()
                self.latest_landmarks = current_processed

    def get_frame(self):
        with self.lock:
            return self.latest_frame

    def get_state(self):
        with self.lock:
            return {"gesture": self.latest_gesture, "fps": self.fps}

    def get_landmarks(self):
        with self.lock:
            return list(self.latest_landmarks)


# Created on startup so the camera only opens when the server actually starts
medical: MedicalSystem | None = None
custom_sign_store: CustomSignStore | None = None

@app.on_event("startup")
async def on_startup():
    global medical, custom_sign_store
    custom_sign_store = CustomSignStore()
    medical = MedicalSystem()
    Thread(target=medical.process_loop, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


def _frame_generator():
    """Sync generator — FastAPI wraps this in a thread automatically."""
    while True:
        frame = medical.get_frame() if medical else None
        if frame:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )
        time.sleep(0.04)   # ~25 fps cap


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/events")
async def events(request: Request):
    async def _sse():
        while True:
            if await request.is_disconnected():
                break
            state = medical.get_state() if medical else {"gesture": "", "fps": 0}
            yield f"data: {json.dumps(state)}\n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(_sse(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────────────────────
# Custom Sign API endpoints (EXISTING — preserved exactly)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/current_landmarks")
def current_landmarks():
    """Returns the latest preprocessed landmark vector (42 floats) for capture."""
    lm = medical.get_landmarks() if medical else []
    return {"landmarks": lm, "has_hand": len(lm) > 0}


@app.post("/custom_sign/save")
async def save_custom_sign(request: Request):
    try:
        data = await request.json()
        name = data.get('name', '').strip().upper()
        icon = data.get('icon', '✋')
        landmarks = data.get('landmarks', [])
        version = data.get('version', 'hand_only')
        if not name:
            return {"success": False, "message": "Sign name is required"}
        if not landmarks:
            return {"success": False, "message": "No hand detected — show your hand clearly to the camera"}
        custom_sign_store.add_sample(name, icon, landmarks, version)
        count = custom_sign_store.sample_count(name)
        return {"success": True, "message": f"'{name}' saved ({count} sample{'s' if count>1 else ''} — more = better accuracy)"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.get("/custom_sign/list")
def list_custom_signs():
    return custom_sign_store.list_signs() if custom_sign_store else {}


@app.delete("/custom_sign/{name}")
def delete_custom_sign(name: str):
    if custom_sign_store:
        custom_sign_store.delete_sign(name.upper())
    return {"success": True}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
