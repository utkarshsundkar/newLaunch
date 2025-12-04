import io
import os
import time
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import tempfile

try:
    import mediapipe as mp
except Exception as e:
    raise RuntimeError("Install mediapipe in your Python env: pip install mediapipe")

# MediaPipe Hands for finger counting
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)

app = FastAPI(title="Pose Server (MediaPipe)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose
# Relax confidences to improve initial detection
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=False,
    enable_segmentation=False,
    min_detection_confidence=0.25,
    min_tracking_confidence=0.25,
)

# Landmark indices for convenience (MediaPipe Pose)
LM = mp.solutions.pose.PoseLandmark


class InferResponse(BaseModel):
    suggested_state: str
    feet_apart: float
    fps: float
    detected: bool
    # Minimal set of landmarks (normalized 0..1) as objects {x,y}
    landmarks: Dict[str, Dict[str, float]]


def compute_metrics(lm_list, h, w):
    LM = mp_pose.PoseLandmark
    def pt(idx):
        lm = lm_list[idx]
        return lm.x, lm.y

    pts = {}
    try:
        pts["l_shoulder"] = pt(LM.LEFT_SHOULDER.value)
        pts["r_shoulder"] = pt(LM.RIGHT_SHOULDER.value)
        pts["l_hip"] = pt(LM.LEFT_HIP.value)
        pts["r_hip"] = pt(LM.RIGHT_HIP.value)
        pts["l_knee"] = pt(LM.LEFT_KNEE.value)
        pts["r_knee"] = pt(LM.RIGHT_KNEE.value)
        pts["l_wrist"] = pt(LM.LEFT_WRIST.value)
        pts["r_wrist"] = pt(LM.RIGHT_WRIST.value)
        pts["l_ankle"] = pt(LM.LEFT_ANKLE.value)
        pts["r_ankle"] = pt(LM.RIGHT_ANKLE.value)
        pts["nose"] = pt(LM.NOSE.value)
    except Exception:
        return None

    def dist(a, b):
        ax, ay = a
        bx, by = b
        return np.hypot(ax - bx, ay - by)

    shoulder_width = max(1e-3, dist(pts["l_shoulder"], pts["r_shoulder"]))
    hip_y = (pts["l_hip"][1] + pts["r_hip"][1]) / 2.0
    head_y = pts["nose"][1]
    wrist_y_mean = (pts["l_wrist"][1] + pts["r_wrist"][1]) / 2.0
    feet_apart = dist(pts["l_ankle"], pts["r_ankle"]) / shoulder_width
    shoulder_y = (pts["l_shoulder"][1] + pts["r_shoulder"][1]) / 2.0
    nose_shoulder_span = max(1e-3, abs(head_y - shoulder_y))

    return {
        "feet_apart": feet_apart,
        "wrist_y_mean": wrist_y_mean,
        "head_y": head_y,
        "hip_y": hip_y,
        "nose_shoulder_span": nose_shoulder_span,
        "pts": pts,
        "shoulder_width": shoulder_width,
    }


def evaluate_state(metrics, open_feet=1.8, close_feet=0.8, open_hands=0.15, close_hands=0.15):
    feet_apart = metrics["feet_apart"]
    wrist_y_mean = metrics["wrist_y_mean"]
    head_y = metrics["head_y"]
    hip_y = metrics["hip_y"]
    nose_shoulder_span = metrics["nose_shoulder_span"]
    hands_up = wrist_y_mean < (head_y - open_hands * nose_shoulder_span)
    hands_down = wrist_y_mean > (hip_y + close_hands * nose_shoulder_span)
    if hands_up and feet_apart >= open_feet:
        return "OPEN"
    if hands_down and feet_apart <= close_feet:
        return "CLOSE"
    return "MID"


@app.post("/infer", response_model=InferResponse)
async def infer(
    file: UploadFile = File(...),
    flip: Optional[int] = Form(default=0),  # set to 1 for front camera mirroring
):
    t0 = time.time()
    data = await file.read()

    # Decode bytes -> BGR
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return InferResponse(suggested_state="NONE", feet_apart=0.0, fps=0.0, detected=False, landmarks={})

    # Downscale for speed if large
    h, w = img.shape[:2]
    max_w = 160
    if w > max_w:
        scale = max_w / float(w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]

    # Optional horizontal flip for front camera
    if flip:
        img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    lm = results.pose_landmarks.landmark if results.pose_landmarks else None
    metrics = None
    if lm is not None:
        metrics = compute_metrics(lm, h, w)

    if metrics is None:
        fps = 1.0 / max(1e-6, time.time() - t0)
        return InferResponse(suggested_state="NONE", feet_apart=0.0, fps=fps, detected=False, landmarks={})

    state = evaluate_state(metrics)
    fps = 1.0 / max(1e-6, time.time() - t0)

    # Return minimal normalized landmarks for overlay if needed
    pts = metrics["pts"]
    def obj(p):
        return {"x": float(p[0]), "y": float(p[1])}
    landmarks = {
        "l_shoulder": obj(pts["l_shoulder"]),
        "r_shoulder": obj(pts["r_shoulder"]),
        "l_hip": obj(pts["l_hip"]),
        "r_hip": obj(pts["r_hip"]),
        "l_wrist": obj(pts["l_wrist"]),
        "r_wrist": obj(pts["r_wrist"]),
        "l_ankle": obj(pts["l_ankle"]),
        "r_ankle": obj(pts["r_ankle"]),
        "nose": obj(pts["nose"]),
    }

    return InferResponse(
        suggested_state=state,
        feet_apart=float(metrics["feet_apart"]),
        fps=fps,
        detected=True,
        landmarks=landmarks,
    )


# ---------------------- Jumping Jacks (server-side) ----------------------
class JacksResponse(BaseModel):
    reps: int
    phase: str
    fps: float
    detected: bool
    landmarks: Dict[str, Dict[str, float]]
    feet_apart: float
    hands_up: bool
    hands_down: bool


# Per-session state
_sessions: Dict[str, Dict[str, int | str]] = {}


def _update_session(session_id: str, suggested: str) -> Tuple[int, str]:
    s = _sessions.get(session_id, {"reps": 0, "phase": "INIT"})
    phase = s["phase"]  # type: ignore
    reps = int(s["reps"])  # type: ignore
    if (phase in ("INIT", "CLOSE")) and suggested == "OPEN":
        phase = "OPEN"
    elif phase == "OPEN" and suggested == "CLOSE":
        phase = "CLOSE"
        reps += 1
    s["phase"], s["reps"] = phase, reps
    _sessions[session_id] = s
    return reps, phase  # type: ignore


@app.post("/jacks", response_model=JacksResponse)
async def jacks(
    file: UploadFile = File(...),
    flip: Optional[int] = Form(default=0),
    session: Optional[str] = Form(default="default"),
):
    t0 = time.time()
    data = await file.read()

    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JacksResponse(reps=0, phase="INIT", fps=0.0, detected=False, landmarks={}, feet_apart=0.0, hands_up=False, hands_down=False)

    h, w = img.shape[:2]
    max_w = 160
    if w > max_w:
        scale = max_w / float(w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]

    if flip:
        img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    lm = results.pose_landmarks.landmark if results.pose_landmarks else None
    if not lm:
        fps = 1.0 / max(1e-6, time.time() - t0)
        session_data = _sessions.get(session or "default", {})
        return JacksResponse(
            reps=int(session_data.get("reps", 0)), 
            phase=str(session_data.get("phase", "INIT")), 
            fps=fps, 
            detected=False, 
            landmarks={},
            feet_apart=0.0,
            hands_up=False,
            hands_down=False
        )

    metrics = compute_metrics(lm, h, w)
    if metrics is None:
        fps = 1.0 / max(1e-6, time.time() - t0)
        session_data = _sessions.get(session or "default", {})
        return JacksResponse(
            reps=int(session_data.get("reps", 0)), 
            phase=str(session_data.get("phase", "INIT")), 
            fps=fps, 
            detected=False, 
            landmarks={},
            feet_apart=0.0,
            hands_up=False,
            hands_down=False
        )
        
    # More permissive thresholds to exit INIT quickly
    state = evaluate_state(metrics, open_feet=1.2, close_feet=0.6, open_hands=0.05, close_hands=0.05)
    reps, phase = _update_session(session or "default", state)
    fps = 1.0 / max(1e-6, time.time() - t0)

    pts = metrics["pts"]
    def obj(p):
        return {"x": float(p[0]), "y": float(p[1])}
    landmarks = {
        "l_shoulder": obj(pts["l_shoulder"]),
        "r_shoulder": obj(pts["r_shoulder"]),
        "l_hip": obj(pts["l_hip"]),
        "r_hip": obj(pts["r_hip"]),
        "l_wrist": obj(pts["l_wrist"]),
        "r_wrist": obj(pts["r_wrist"]),
        "l_ankle": obj(pts["l_ankle"]),
        "r_ankle": obj(pts["r_ankle"]),
        "nose": obj(pts["nose"]),
    }

    # Debug flags
    feet_apart = float(metrics["feet_apart"])
    wrist_y_mean = metrics["wrist_y_mean"]
    head_y = metrics["head_y"]
    hip_y = metrics["hip_y"]
    nose_shoulder_span = metrics["nose_shoulder_span"]
    hands_up = wrist_y_mean < (head_y - 0.05 * nose_shoulder_span)
    hands_down = wrist_y_mean > (hip_y + 0.05 * nose_shoulder_span)

    return JacksResponse(
        reps=reps,
        phase=phase,
        fps=fps,
        detected=True,
        landmarks=landmarks,
        feet_apart=feet_apart,
        hands_up=bool(hands_up),
        hands_down=bool(hands_down),
    )


class FingersResponse(BaseModel):
    detected: bool
    fingers: int
    handedness: str | None
    fps: float


def count_fingers(landmarks, handedness_label: str) -> int:
    # landmarks: list of 21 points with x,y in normalized coords
    # Heuristic: For index/middle/ring/pinky, tip is above PIP (smaller y) when extended.
    # For thumb, use x direction based on handedness.
    TIP = [mp_hands.HandLandmark.THUMB_TIP,
           mp_hands.HandLandmark.INDEX_FINGER_TIP,
           mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
           mp_hands.HandLandmark.RING_FINGER_TIP,
           mp_hands.HandLandmark.PINKY_TIP]
    PIP = [mp_hands.HandLandmark.THUMB_IP,
           mp_hands.HandLandmark.INDEX_FINGER_PIP,
           mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
           mp_hands.HandLandmark.RING_FINGER_PIP,
           mp_hands.HandLandmark.PINKY_PIP]

    # y grows downward in image coords
    count = 0
    # Non-thumb fingers
    for i in [1, 2, 3, 4]:
        if landmarks[TIP[i]].y < landmarks[PIP[i]].y:
            count += 1

    # Thumb: compare x based on handedness
    if handedness_label.lower().startswith('right'):
        if landmarks[TIP[0]].x < landmarks[PIP[0]].x:
            count += 1
    else:  # left
        if landmarks[TIP[0]].x > landmarks[PIP[0]].x:
            count += 1
    return count


@app.post("/fingers", response_model=FingersResponse)
async def fingers(
    file: UploadFile = File(...),
    flip: Optional[int] = Form(default=0),
):
    t0 = time.time()
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return FingersResponse(detected=False, fingers=0, handedness=None, fps=0.0)

    h, w = img.shape[:2]
    max_w = 320
    if w > max_w:
        scale = max_w / float(w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    if flip:
        img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    fps = 1.0 / max(1e-6, time.time() - t0)

    if not result.multi_hand_landmarks or not result.multi_handedness:
        return FingersResponse(detected=False, fingers=0, handedness=None, fps=fps)

    # Use first hand
    hand_lm = result.multi_hand_landmarks[0].landmark
    hand_label = result.multi_handedness[0].classification[0].label  # 'Left' or 'Right'
    fingers = count_fingers(hand_lm, hand_label)
    return FingersResponse(detected=True, fingers=int(fingers), handedness=hand_label, fps=fps)


# ---------------------- Offline Video Analysis ----------------------
class AnalyzeResponse(BaseModel):
    reps: int
    frames: int
    duration_s: float
    processed_fps: float
    reps_perfect: int
    reps_wrong: int
    diagnostics: list | None = None


def _draw_overlay(img, pts: Dict[str, Tuple[float, float]], messages: list[str]):
    # pts are normalized; draw simple skeleton and messages
    h, w = img.shape[:2]
    def denorm(p):
        return int(max(0, min(1, p[0])) * w), int(max(0, min(1, p[1])) * h)
    # Key connections
    lines = [
        ("l_shoulder", "r_shoulder"),
        ("l_shoulder", "l_hip"),
        ("r_shoulder", "r_hip"),
        ("l_hip", "r_hip"),
        ("l_shoulder", "l_wrist"),
        ("r_shoulder", "r_wrist"),
        ("l_hip", "l_ankle"),
        ("r_hip", "r_ankle"),
    ]
    for a, b in lines:
        if a in pts and b in pts:
            pa = denorm(pts[a]); pb = denorm(pts[b])
            cv2.line(img, pa, pb, (0, 255, 0), 2)
    for k, p in pts.items():
        cv2.circle(img, denorm(p), 3, (255, 0, 0), -1)
    # Put messages at top-left
    y = 20
    for m in messages:
        cv2.putText(img, m, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        y += 22
    # Encode to JPEG b64
    ok, buf = cv2.imencode('.jpg', img)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode('ascii')


def _count_jacks_in_video(path: str, flip: bool = False, sample_stride: int = 2) -> Tuple[int, int, float, int, int, list]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0, 0, 0.0, 0, 0, []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = total_frames / fps if fps > 0 else 0.0

    reps = 0
    reps_perfect = 0
    reps_wrong = 0
    phase = "INIT"
    processed = 0
    idx = 0
    # Track quality across an OPEN->CLOSE cycle
    open_ok = False
    close_ok = False
    diagnostics: list = []
    # store a representative frame and pts for current rep
    last_open_snapshot = None  # (frame_bgr, pts)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if sample_stride > 1 and (idx % sample_stride) != 0:
            idx += 1
            continue
        idx += 1
        if flip:
            frame = cv2.flip(frame, 1)

        # Resize to speed up
        h, w = frame.shape[:2]
        max_w = 160
        if w > max_w:
            scale = max_w / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        lm = results.pose_landmarks.landmark if results.pose_landmarks else None
        if not lm:
            continue

        metrics = compute_metrics(lm, h, w)
        if metrics is None:
            continue
            
        state = evaluate_state(metrics, open_feet=1.2, close_feet=0.6, open_hands=0.05, close_hands=0.05)

        # Strict quality checks
        feet_apart = float(metrics.get("feet_apart", 0.0)) if metrics else 0.0
        wrist_y_mean = metrics.get("wrist_y_mean", 0.0) if metrics else 0.0
        head_y = metrics.get("head_y", 0.0) if metrics else 0.0
        hip_y = metrics.get("hip_y", 1.0) if metrics else 1.0
        span = metrics.get("nose_shoulder_span", 0.3) if metrics else 0.3
        hands_up_strict = wrist_y_mean < (head_y - 0.10 * span)
        hands_down_strict = wrist_y_mean > (hip_y + 0.08 * span)
        feet_open_strict = feet_apart >= 1.4
        feet_close_strict = feet_apart <= 0.5

        # Simple local state machine (not using session)
        if (phase in ("INIT", "CLOSE")) and state == "OPEN":
            phase = "OPEN"
            # capture open quality at peak
            open_ok = hands_up_strict and feet_open_strict
            close_ok = False
            # snapshot frame and pts for diagnostics
            pts = metrics["pts"]
            last_open_snapshot = (frame.copy(), pts)
        elif phase == "OPEN" and state == "CLOSE":
            reps += 1
            # capture close quality at finish
            close_ok = hands_down_strict and feet_close_strict
            if open_ok and close_ok:
                reps_perfect += 1
            else:
                reps_wrong += 1
                # build description and annotated image using stored OPEN snapshot if available
                issues = []
                if not hands_up_strict:
                    issues.append("Hands not high enough at OPEN")
                if not feet_open_strict:
                    issues.append("Feet not wide enough at OPEN")
                if not hands_down_strict:
                    issues.append("Hands not down at CLOSE")
                if not feet_close_strict:
                    issues.append("Feet not together at CLOSE")
                img_b64 = None
                if last_open_snapshot is not None:
                    snap_img, snap_pts = last_open_snapshot
                    img_b64 = _draw_overlay(snap_img, snap_pts, issues)
                diagnostics.append({
                    "rep_index": int(reps),
                    "description": "; ".join(issues) if issues else "Form deviation",
                    "image_b64": img_b64,
                })
            phase = "CLOSE"

        processed += 1

    cap.release()
    proc_fps = processed / duration if duration > 0 else 0.0
    return reps, processed, duration, reps_perfect, reps_wrong, diagnostics


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_video(file: UploadFile = File(...), flip: Optional[int] = Form(default=0)):
    # Persist upload to a temp file for OpenCV
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "video.mp4")[1] or ".mp4") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        reps, processed, duration, reps_perfect, reps_wrong, diagnostics = _count_jacks_in_video(tmp_path, flip=bool(flip), sample_stride=2)
        proc_fps = processed / duration if duration > 0 else 0.0
        return AnalyzeResponse(
            reps=int(reps),
            frames=int(processed),
            duration_s=float(duration),
            processed_fps=float(proc_fps),
            reps_perfect=int(reps_perfect),
            reps_wrong=int(reps_wrong),
            diagnostics=diagnostics,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------- WebSocket: Jumping Jacks ----------------------
@app.websocket("/ws/jacks")
async def ws_jacks(websocket: WebSocket):
    await websocket.accept()
    print("[WS] /ws/jacks connected from:", websocket.client)
    # Query params: flip, session
    params = websocket.query_params
    flip_q = params.get("flip", "0")
    session_id = params.get("session", "default")
    flip_flag = 1 if str(flip_q) == "1" else 0

    try:
        while True:
            # Expect text JSON: {"jpg_b64": "..."}
            msg = await websocket.receive_json()
            t0 = time.time()
            b64 = msg.get("jpg_b64")
            if not b64:
                await websocket.send_json({"error": "missing jpg_b64"})
                continue

            # Decode JPEG
            try:
                img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                await websocket.send_json({"error": "decode_failed"})
                continue
            if img is None:
                await websocket.send_json({"error": "imdecode_failed"})
                continue

            h, w = img.shape[:2]
            max_w = 160
            if w > 0:
                scale = 160.0 / max(1.0, float(w))
                if scale < 1.0:
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
                    h, w = img.shape[:2]

            if flip_flag:
                img = cv2.flip(img, 1)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            lm = results.pose_landmarks.landmark if results.pose_landmarks else None
            if not lm:
                fps = 1.0 / max(1e-6, time.time() - t0)
                await websocket.send_json({
                    "reps": int(_sessions.get(session_id, {}).get("reps", 0)),
                    "phase": str(_sessions.get(session_id, {}).get("phase", "INIT")),
                    "fps": fps,
                    "detected": False,
                    "landmarks": {},
                })
                continue

            metrics = compute_metrics(lm, h, w)
            state = evaluate_state(metrics, open_feet=1.2, close_feet=0.6, open_hands=0.05, close_hands=0.05)
            reps, phase = _update_session(session_id, state)
            fps = 1.0 / max(1e-6, time.time() - t0)

            pts = metrics["pts"]
            def obj(p):
                return {"x": float(p[0]), "y": float(p[1])}
            landmarks = {
                "l_shoulder": obj(pts["l_shoulder"]),
                "r_shoulder": obj(pts["r_shoulder"]),
                "l_hip": obj(pts["l_hip"]),
                "r_hip": obj(pts["r_hip"]),
                "l_wrist": obj(pts["l_wrist"]),
                "r_wrist": obj(pts["r_wrist"]),
                "l_ankle": obj(pts["l_ankle"]),
                "r_ankle": obj(pts["r_ankle"]),
                "nose": obj(pts["nose"]),
            }

            await websocket.send_json({
                "reps": reps,
                "phase": phase,
                "fps": fps,
                "detected": True,
                "landmarks": landmarks,
            })

    except WebSocketDisconnect:
        print("[WS] /ws/jacks disconnected:", websocket.client)
        return


# ---------------------- WebSocket: Pushups ----------------------
@app.websocket("/ws/pushups")
async def ws_pushups(websocket: WebSocket):
    await websocket.accept()
    print("[WS] /ws/pushups connected from:", websocket.client)
    # Query params: flip, session
    params = websocket.query_params
    flip_q = params.get("flip", "0")
    session_id = params.get("session", "default")
    flip_flag = 1 if str(flip_q) == "1" else 0

    try:
        while True:
            # Expect text JSON: {"jpg_b64": "..."}
            msg = await websocket.receive_json()
            t0 = time.time()
            b64 = msg.get("jpg_b64")
            if not b64:
                await websocket.send_json({"error": "missing jpg_b64"})
                continue

            # Decode JPEG
            try:
                img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                await websocket.send_json({"error": "decode_failed"})
                continue
            if img is None:
                await websocket.send_json({"error": "imdecode_failed"})
                continue

            h, w = img.shape[:2]
            max_w = 160
            if w > 0:
                scale = 160.0 / max(1.0, float(w))
                if scale < 1.0:
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
                    h, w = img.shape[:2]

            if flip_flag:
                img = cv2.flip(img, 1)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            lm = results.pose_landmarks.landmark if results.pose_landmarks else None
            if not lm:
                fps = 1.0 / max(1e-6, time.time() - t0)
                session_data = _pushup_sessions.get(session_id, {})
                await websocket.send_json({
                    "reps": int(session_data.get("reps", 0)),
                    "phase": str(session_data.get("phase", "INIT")),
                    "fps": fps,
                    "detected": False,
                    "landmarks": {},
                    "body_angle": 0.0,
                    "arms_angle": 0.0,
                    "is_down": False,
                    "is_up": False,
                })
                continue

            # Extended metrics for pushups
            def pt(idx):
                lm_pt = lm[idx]
                return lm_pt.x, lm_pt.y

            pts = {}
            try:
                pts["l_shoulder"] = pt(LM.LEFT_SHOULDER.value)
                pts["r_shoulder"] = pt(LM.RIGHT_SHOULDER.value)
                pts["l_hip"] = pt(LM.LEFT_HIP.value)
                pts["r_hip"] = pt(LM.RIGHT_HIP.value)
                pts["l_wrist"] = pt(LM.LEFT_WRIST.value)
                pts["r_wrist"] = pt(LM.RIGHT_WRIST.value)
                pts["l_ankle"] = pt(LM.LEFT_ANKLE.value)
                pts["r_ankle"] = pt(LM.RIGHT_ANKLE.value)
                pts["l_elbow"] = pt(LM.LEFT_ELBOW.value)
                pts["r_elbow"] = pt(LM.RIGHT_ELBOW.value)
                pts["nose"] = pt(LM.NOSE.value)
            except Exception:
                fps = 1.0 / max(1e-6, time.time() - t0)
                session_data = _pushup_sessions.get(session_id, {})
                await websocket.send_json({
                    "reps": int(session_data.get("reps", 0)),
                    "phase": str(session_data.get("phase", "INIT")),
                    "fps": fps,
                    "detected": False,
                    "landmarks": {},
                    "body_angle": 0.0,
                    "arms_angle": 0.0,
                    "is_down": False,
                    "is_up": False,
                })
                continue

            def dist(a, b):
                ax, ay = a
                bx, by = b
                return np.hypot(ax - bx, ay - by)

            shoulder_width = max(1e-3, dist(pts["l_shoulder"], pts["r_shoulder"]))

            # Calculate body angle for pushup detection
            shoulder_y = (pts["l_shoulder"][1] + pts["r_shoulder"][1]) / 2.0
            hip_y = (pts["l_hip"][1] + pts["r_hip"][1]) / 2.0
            ankle_y = (pts["l_ankle"][1] + pts["r_ankle"][1]) / 2.0

            shoulder_ankle_mid_y = (shoulder_y + ankle_y) / 2.0
            body_deviation = abs(hip_y - shoulder_ankle_mid_y)
            body_angle = body_deviation / shoulder_width

            # Calculate arm angle (simplified)
            shoulder_hip_dist = abs(shoulder_y - hip_y)
            arms_angle = shoulder_hip_dist / shoulder_width

            metrics = {
                "pts": pts,
                "shoulder_width": shoulder_width,
                "body_angle": body_angle,
                "arms_angle": arms_angle
            }

            # Evaluate pushup state
            state = evaluate_pushup_state(metrics)
            reps, phase = _update_pushup_session(session_id, state)
            fps = 1.0 / max(1e-6, time.time() - t0)

            def obj(p):
                return {"x": float(p[0]), "y": float(p[1])}
            landmarks = {
                "l_shoulder": obj(pts["l_shoulder"]),
                "r_shoulder": obj(pts["r_shoulder"]),
                "l_hip": obj(pts["l_hip"]),
                "r_hip": obj(pts["r_hip"]),
                "l_wrist": obj(pts["l_wrist"]),
                "r_wrist": obj(pts["r_wrist"]),
                "l_ankle": obj(pts["l_ankle"]),
                "r_ankle": obj(pts["r_ankle"]),
                "l_elbow": obj(pts["l_elbow"]),
                "r_elbow": obj(pts["r_elbow"]),
                "nose": obj(pts["nose"]),
            }

            # Debug flags
            is_down = state == "DOWN"
            is_up = state == "UP"

            await websocket.send_json({
                "reps": reps,
                "phase": phase,
                "fps": fps,
                "detected": True,
                "landmarks": landmarks,
                "body_angle": float(body_angle),
                "arms_angle": float(arms_angle),
                "is_down": bool(is_down),
                "is_up": bool(is_up),
            })

    except WebSocketDisconnect:
        print("[WS] /ws/pushups disconnected:", websocket.client)
        return


# ---------------------- Pushups (server-side) ----------------------
class PushupResponse(BaseModel):
    reps: int
    phase: str
    fps: float
    detected: bool
    landmarks: Dict[str, Dict[str, float]]
    body_angle: float
    arms_angle: float
    is_down: bool
    is_up: bool


# Per-session state for pushups
_pushup_sessions: Dict[str, Dict[str, int | str]] = {}


def _count_pushups_in_video(path: str, flip: bool = False, sample_stride: int = 2) -> Tuple[int, int, float, int, dict]:
    """Process entire video and count pushups"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0, 0, 0.0, 0, {}
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = total_frames / fps if fps > 0 else 0.0
    
    reps = 0
    phase = "INIT"
    processed = 0
    idx = 0
    
    # For diagnostics, track some stats
    diagnostics = {
        "max_body_angle": 0.0,
        "min_body_angle": 1.0,
        "max_arm_angle": 0.0,
        "min_arm_angle": 1.0
    }
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if sample_stride > 1 and (idx % sample_stride) != 0:
            idx += 1
            continue
        idx += 1
        if flip:
            frame = cv2.flip(frame, 1)
            
        # Resize to speed up
        h, w = frame.shape[:2]
        max_w = 160
        if w > max_w:
            scale = max_w / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            h, w = frame.shape[:2]
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        lm = results.pose_landmarks.landmark if results.pose_landmarks else None
        if not lm:
            continue
            
        # Extract points
        def pt(idx):
            lm_pt = lm[idx]
            return lm_pt.x, lm_pt.y
            
        pts = {}
        try:
            pts["l_shoulder"] = pt(LM.LEFT_SHOULDER.value)
            pts["r_shoulder"] = pt(LM.RIGHT_SHOULDER.value)
            pts["l_hip"] = pt(LM.LEFT_HIP.value)
            pts["r_hip"] = pt(LM.RIGHT_HIP.value)
            pts["l_wrist"] = pt(LM.LEFT_WRIST.value)
            pts["r_wrist"] = pt(LM.RIGHT_WRIST.value)
            pts["l_ankle"] = pt(LM.LEFT_ANKLE.value)
            pts["r_ankle"] = pt(LM.RIGHT_ANKLE.value)
            pts["l_elbow"] = pt(LM.LEFT_ELBOW.value)
            pts["r_elbow"] = pt(LM.RIGHT_ELBOW.value)
            pts["nose"] = pt(LM.NOSE.value)
        except Exception:
            continue
            
        def dist(a, b):
            ax, ay = a
            bx, by = b
            return np.hypot(ax - bx, ay - by)
            
        shoulder_width = max(1e-3, dist(pts["l_shoulder"], pts["r_shoulder"]))
        
        # Calculate body angle for pushup detection
        shoulder_y = (pts["l_shoulder"][1] + pts["r_shoulder"][1]) / 2.0
        hip_y = (pts["l_hip"][1] + pts["r_hip"][1]) / 2.0
        ankle_y = (pts["l_ankle"][1] + pts["r_ankle"][1]) / 2.0
        
        shoulder_ankle_mid_y = (shoulder_y + ankle_y) / 2.0
        body_deviation = abs(hip_y - shoulder_ankle_mid_y)
        body_angle = body_deviation / shoulder_width
        
        # Calculate arm angle (simplified)
        shoulder_hip_dist = abs(shoulder_y - hip_y)
        arms_angle = shoulder_hip_dist / shoulder_width
        
        # Update diagnostics
        diagnostics["max_body_angle"] = max(diagnostics["max_body_angle"], body_angle)
        diagnostics["min_body_angle"] = min(diagnostics["min_body_angle"], body_angle)
        diagnostics["max_arm_angle"] = max(diagnostics["max_arm_angle"], arms_angle)
        diagnostics["min_arm_angle"] = min(diagnostics["min_arm_angle"], arms_angle)
        
        metrics = {
            "pts": pts,
            "shoulder_width": shoulder_width,
            "body_angle": body_angle,
            "arms_angle": arms_angle
        }
        
        # Evaluate pushup state
        state = evaluate_pushup_state(metrics)
        
        # Simple state machine for counting
        if (phase in ("INIT", "UP")) and state == "DOWN":
            phase = "DOWN"
        elif phase == "DOWN" and state == "UP":
            phase = "UP"
            reps += 1
            
        processed += 1
        
    cap.release()
    return reps, processed, duration, 0, diagnostics  # 0 for reps_wrong (not implemented yet)


def _update_pushup_session(session_id: str, suggested: str) -> Tuple[int, str]:
    s = _pushup_sessions.get(session_id, {"reps": 0, "phase": "INIT"})
    phase = str(s["phase"])  # type: ignore
    reps = int(s["reps"])  # type: ignore
    if (phase in ("INIT", "UP")) and suggested == "DOWN":
        phase = "DOWN"
    elif phase == "DOWN" and suggested == "UP":
        phase = "UP"
        reps += 1
    s["phase"], s["reps"] = phase, reps
    _pushup_sessions[session_id] = s
    return reps, phase  # type: ignore


def evaluate_pushup_state(metrics):
    """Evaluate pushup state based on body angles and positions"""
    # Calculate body angle (shoulder to ankle line relative to horizontal)
    shoulder_y = (metrics["pts"]["l_shoulder"][1] + metrics["pts"]["r_shoulder"][1]) / 2.0
    hip_y = (metrics["pts"]["l_hip"][1] + metrics["pts"]["r_hip"][1]) / 2.0
    ankle_y = (metrics["pts"]["l_ankle"][1] + metrics["pts"]["r_ankle"][1]) / 2.0
    
    # Calculate if body is straight (hips aligned with shoulder-ankle line)
    shoulder_ankle_mid_y = (shoulder_y + ankle_y) / 2.0
    body_deviation = abs(hip_y - shoulder_ankle_mid_y)
    body_angle = body_deviation / metrics["shoulder_width"]  # Normalize by shoulder width
    
    # Calculate arm bend angle (using shoulder, elbow, wrist)
    # For simplicity, we'll use a heuristic based on shoulder and hip position
    shoulder_hip_dist = abs(shoulder_y - hip_y)
    
    # Determine if in down or up position
    # Down position: body straight, arms bent (shoulder closer to hip)
    # Up position: body straight, arms extended (shoulder farther from hip)
    
    # Even more permissive thresholds to improve detection
    is_down = shoulder_hip_dist < metrics["shoulder_width"] * 0.7  # Even more relaxed threshold
    is_up = shoulder_hip_dist > metrics["shoulder_width"] * 0.3   # Even more relaxed threshold
    
    # Even more permissive body angle check
    if is_down and body_angle < 0.7:  # Much more permissive body straightness check
        return "DOWN"
    elif is_up and body_angle < 0.7:  # Much more permissive body straightness check
        return "UP"
    return "MID"


@app.post("/pushups", response_model=PushupResponse)
async def pushups(
    file: UploadFile = File(...),
    flip: Optional[int] = Form(default=0),
    session: Optional[str] = Form(default="default"),
):
    # For single frame analysis (like real-time), use existing logic
    # But for video analysis, we should process the entire video
    # Let's check if this is a video file and process accordingly
    
    # Persist upload to a temp file for OpenCV
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "video.mp4")[1] or ".mp4") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        reps, processed, duration, reps_wrong, diagnostics = _count_pushups_in_video(tmp_path, flip=bool(flip), sample_stride=2)
        proc_fps = processed / duration if duration > 0 else 0.0
        
        # Return a summary response similar to jumping jacks analysis
        return PushupResponse(
            reps=reps,
            phase="COMPLETE",  # Indicate video analysis is complete
            fps=proc_fps,
            detected=True,
            landmarks={},  # Not applicable for video summary
            body_angle=diagnostics.get("max_body_angle", 0.0),
            arms_angle=diagnostics.get("max_arm_angle", 0.0),
            is_down=False,
            is_up=False
        )
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass


# ---------------------- Diamond Pushups (server-side) ----------------------
class DiamondPushupResponse(BaseModel):
    reps: int
    phase: str
    fps: float
    detected: bool
    landmarks: Dict[str, Dict[str, float]]
    body_angle: float
    arms_angle: float
    hand_distance: float
    is_down: bool
    is_up: bool


@app.post("/diamond_pushups", response_model=DiamondPushupResponse)
async def diamond_pushups(
    file: UploadFile = File(...),
    flip: Optional[int] = Form(default=0),
    session: Optional[str] = Form(default="default"),
):
    # Persist upload to a temp file for OpenCV
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "video.mp4")[1] or ".mp4") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        reps, processed, duration, reps_wrong, diagnostics = _count_diamond_pushups_in_video(tmp_path, flip=bool(flip), sample_stride=2)
        proc_fps = processed / duration if duration > 0 else 0.0
        
        # Return a summary response similar to jumping jacks analysis
        return DiamondPushupResponse(
            reps=reps,
            phase="COMPLETE",  # Indicate video analysis is complete
            fps=proc_fps,
            detected=True,
            landmarks={},  # Not applicable for video summary
            body_angle=diagnostics.get("max_body_angle", 0.0),
            arms_angle=diagnostics.get("max_arm_angle", 0.0),
            hand_distance=diagnostics.get("min_hand_distance", 0.0),  # For diamond pushups, min distance is more relevant
            is_down=False,
            is_up=False
        )
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass


def _count_diamond_pushups_in_video(path: str, flip: bool = False, sample_stride: int = 2) -> Tuple[int, int, float, int, dict]:
    """Process entire video and count diamond pushups specifically"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0, 0, 0.0, 0, {}
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = total_frames / fps if fps > 0 else 0.0
    
    reps = 0
    phase = "INIT"
    processed = 0
    idx = 0
    
    # For diagnostics, track some stats
    diagnostics = {
        "max_body_angle": 0.0,
        "min_body_angle": 1.0,
        "max_arm_angle": 0.0,
        "min_arm_angle": 1.0,
        "max_hand_distance": 0.0,
        "min_hand_distance": 1.0
    }
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if sample_stride > 1 and (idx % sample_stride) != 0:
            idx += 1
            continue
        idx += 1
        if flip:
            frame = cv2.flip(frame, 1)
            
        # Resize to speed up
        h, w = frame.shape[:2]
        max_w = 160
        if w > max_w:
            scale = max_w / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            h, w = frame.shape[:2]
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        lm = results.pose_landmarks.landmark if results.pose_landmarks else None
        if not lm:
            continue
            
        # Extract points
        def pt(idx):
            lm_pt = lm[idx]
            return lm_pt.x, lm_pt.y
            
        pts = {}
        try:
            pts["l_shoulder"] = pt(LM.LEFT_SHOULDER.value)
            pts["r_shoulder"] = pt(LM.RIGHT_SHOULDER.value)
            pts["l_hip"] = pt(LM.LEFT_HIP.value)
            pts["r_hip"] = pt(LM.RIGHT_HIP.value)
            pts["l_wrist"] = pt(LM.LEFT_WRIST.value)
            pts["r_wrist"] = pt(LM.RIGHT_WRIST.value)
            pts["l_ankle"] = pt(LM.LEFT_ANKLE.value)
            pts["r_ankle"] = pt(LM.RIGHT_ANKLE.value)
            pts["l_elbow"] = pt(LM.LEFT_ELBOW.value)
            pts["r_elbow"] = pt(LM.RIGHT_ELBOW.value)
            pts["nose"] = pt(LM.NOSE.value)
        except Exception:
            continue
            
        def dist(a, b):
            ax, ay = a
            bx, by = b
            return np.hypot(ax - bx, ay - by)
            
        shoulder_width = max(1e-3, dist(pts["l_shoulder"], pts["r_shoulder"]))
        # Calculate distance between wrists for diamond pushup detection
        wrist_distance = dist(pts["l_wrist"], pts["r_wrist"])
        
        # Calculate body angle for pushup detection
        shoulder_y = (pts["l_shoulder"][1] + pts["r_shoulder"][1]) / 2.0
        hip_y = (pts["l_hip"][1] + pts["r_hip"][1]) / 2.0
        ankle_y = (pts["l_ankle"][1] + pts["r_ankle"][1]) / 2.0
        
        shoulder_ankle_mid_y = (shoulder_y + ankle_y) / 2.0
        body_deviation = abs(hip_y - shoulder_ankle_mid_y)
        body_angle = body_deviation / shoulder_width
        
        # Calculate arm angle (simplified)
        shoulder_hip_dist = abs(shoulder_y - hip_y)
        arms_angle = shoulder_hip_dist / shoulder_width
        
        # Update diagnostics
        diagnostics["max_body_angle"] = max(diagnostics["max_body_angle"], body_angle)
        diagnostics["min_body_angle"] = min(diagnostics["min_body_angle"], body_angle)
        diagnostics["max_arm_angle"] = max(diagnostics["max_arm_angle"], arms_angle)
        diagnostics["min_arm_angle"] = min(diagnostics["min_arm_angle"], arms_angle)
        diagnostics["max_hand_distance"] = max(diagnostics["max_hand_distance"], wrist_distance)
        diagnostics["min_hand_distance"] = min(diagnostics["min_hand_distance"], wrist_distance)
        
        metrics = {
            "pts": pts,
            "shoulder_width": shoulder_width,
            "body_angle": body_angle,
            "arms_angle": arms_angle,
            "wrist_distance": wrist_distance
        }
        
        # Evaluate diamond pushup state
        state = evaluate_diamond_pushup_state(metrics)
        
        # Simple state machine for counting
        if (phase in ("INIT", "UP")) and state == "DOWN":
            phase = "DOWN"
        elif phase == "DOWN" and state == "UP":
            phase = "UP"
            reps += 1
            
        processed += 1
        
    cap.release()
    return reps, processed, duration, 0, diagnostics  # 0 for reps_wrong (not implemented yet)


def evaluate_diamond_pushup_state(metrics):
    """Evaluate diamond pushup state based on body angles and hand positioning"""
    # Calculate body angle (shoulder to ankle line relative to horizontal)
    shoulder_y = (metrics["pts"]["l_shoulder"][1] + metrics["pts"]["r_shoulder"][1]) / 2.0
    hip_y = (metrics["pts"]["l_hip"][1] + metrics["pts"]["r_hip"][1]) / 2.0
    ankle_y = (metrics["pts"]["l_ankle"][1] + metrics["pts"]["r_ankle"][1]) / 2.0
    
    # Calculate if body is straight (hips aligned with shoulder-ankle line)
    shoulder_ankle_mid_y = (shoulder_y + ankle_y) / 2.0
    body_deviation = abs(hip_y - shoulder_ankle_mid_y)
    body_angle = body_deviation / metrics["shoulder_width"]  # Normalize by shoulder width
    
    # Calculate arm bend angle (using shoulder, elbow, wrist)
    shoulder_hip_dist = abs(shoulder_y - hip_y)
    
    # Check if hands are close together (diamond pushup characteristic)
    # In diamond pushups, wrist distance should be much smaller than shoulder width
    wrist_distance = metrics["wrist_distance"]
    shoulder_width = metrics["shoulder_width"]
    hands_close = wrist_distance < shoulder_width * 0.4  # Hands should be less than 40% of shoulder width apart
    
    # Determine if in down or up position
    # Down position: body straight, arms bent (shoulder closer to hip)
    # Up position: body straight, arms extended (shoulder farther from hip)
    
    # Even more permissive thresholds to improve detection
    is_down = shoulder_hip_dist < shoulder_width * 0.7  # Even more relaxed threshold
    is_up = shoulder_hip_dist > shoulder_width * 0.3   # Even more relaxed threshold
    
    # Even more permissive body angle check
    if is_down and body_angle < 0.7 and hands_close:  # Much more permissive body straightness check and hands must be close
        return "DOWN"
    elif is_up and body_angle < 0.7 and hands_close:  # Much more permissive body straightness check and hands must be close
        return "UP"
    return "MID"

# ---------------------- Plank (server-side) ----------------------
class PlankResponse(BaseModel):
    duration_perfect: float
    duration_imperfect: float
    is_correct: bool
    feedback: str
    fps: float
    detected: bool
    landmarks: Dict[str, Dict[str, float]]
    body_angle: float
    hip_deviation: float


# Per-session state for plank
# Stores: {"start_time": float, "perfect_time": float, "imperfect_time": float, "last_update": float}
_plank_sessions: Dict[str, Dict[str, float | str]] = {}


def evaluate_plank_state(metrics):
    """
    Evaluate plank state based on body alignment.
    Assumes side view.
    Check if shoulder, hip, and ankle are collinear.
    """
    pts = metrics["pts"]
    
    # Average left and right for robustness (or use the visible side)
    shoulder_y = (pts["l_shoulder"][1] + pts["r_shoulder"][1]) / 2.0
    hip_y = (pts["l_hip"][1] + pts["r_hip"][1]) / 2.0
    ankle_y = (pts["l_ankle"][1] + pts["r_ankle"][1]) / 2.0
    
    shoulder_x = (pts["l_shoulder"][0] + pts["r_shoulder"][0]) / 2.0
    hip_x = (pts["l_hip"][0] + pts["r_hip"][0]) / 2.0
    ankle_x = (pts["l_ankle"][0] + pts["r_ankle"][0]) / 2.0

    # Calculate deviation of hip from the line connecting shoulder and ankle
    # Line equation from shoulder (x1, y1) to ankle (x2, y2):
    # (y - y1) = m * (x - x1)
    # m = (y2 - y1) / (x2 - x1)
    # Expected hip y = y1 + m * (hip_x - x1)
    
    if abs(ankle_x - shoulder_x) < 1e-3:
        # Vertical body? Unlikely for plank, but handle div by zero
        expected_hip_y = hip_y 
    else:
        m = (ankle_y - shoulder_y) / (ankle_x - shoulder_x)
        expected_hip_y = shoulder_y + m * (hip_x - shoulder_x)
    
    # Deviation in Y (vertical)
    # Normalized by shoulder width to be scale invariant
    shoulder_width = metrics["shoulder_width"]
    hip_deviation = (hip_y - expected_hip_y) / shoulder_width
    
    # Thresholds
    # If hip is too low (positive deviation in image coords usually, as y increases downwards)
    # If hip is too high (negative deviation)
    
    # Note: In image coords, y increases downwards.
    # If hip is physically lower than line, y is larger -> positive deviation
    # If hip is physically higher (pike), y is smaller -> negative deviation
    
    feedback = "Good"
    is_correct = True
    
    if hip_deviation > 0.2:
        feedback = "Raise hips"
        is_correct = False
    elif hip_deviation < -0.2:
        feedback = "Lower hips"
        is_correct = False
        
    return is_correct, feedback, hip_deviation


def _update_plank_session(session_id: str, is_correct: bool) -> Tuple[float, float]:
    now = time.time()
    s = _plank_sessions.get(session_id, {
        "start_time": now, 
        "perfect_time": 0.0, 
        "imperfect_time": 0.0, 
        "last_update": now
    })
    
    last_update = float(s["last_update"])
    dt = now - last_update
    
    # Cap dt to avoid huge jumps if requests are sparse
    if dt > 1.0: 
        dt = 0.0
        
    if is_correct:
        s["perfect_time"] = float(s["perfect_time"]) + dt
    else:
        s["imperfect_time"] = float(s["imperfect_time"]) + dt
        
    s["last_update"] = now
    _plank_sessions[session_id] = s
    
    return float(s["perfect_time"]), float(s["imperfect_time"])


@app.post("/plank", response_model=PlankResponse)
async def plank(
    file: UploadFile = File(...),
    flip: Optional[int] = Form(default=0),
    session: Optional[str] = Form(default="default"),
):
    t0 = time.time()
    data = await file.read()
    
    # Decode
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return PlankResponse(
            duration_perfect=0.0, duration_imperfect=0.0, is_correct=False, 
            feedback="No image", fps=0.0, detected=False, landmarks={}, 
            body_angle=0.0, hip_deviation=0.0
        )

    h, w = img.shape[:2]
    max_w = 160
    if w > max_w:
        scale = max_w / float(w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]

    if flip:
        img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    lm = results.pose_landmarks.landmark if results.pose_landmarks else None
    
    if not lm:
        fps = 1.0 / max(1e-6, time.time() - t0)
        s = _plank_sessions.get(session or "default", {"perfect_time": 0.0, "imperfect_time": 0.0})
        return PlankResponse(
            duration_perfect=float(s["perfect_time"]), 
            duration_imperfect=float(s["imperfect_time"]), 
            is_correct=False, 
            feedback="No pose detected", 
            fps=fps, 
            detected=False, 
            landmarks={}, 
            body_angle=0.0, 
            hip_deviation=0.0
        )

    metrics = compute_metrics(lm, h, w)
    if metrics is None:
        fps = 1.0 / max(1e-6, time.time() - t0)
        s = _plank_sessions.get(session or "default", {"perfect_time": 0.0, "imperfect_time": 0.0})
        return PlankResponse(
            duration_perfect=float(s["perfect_time"]), 
            duration_imperfect=float(s["imperfect_time"]), 
            is_correct=False, 
            feedback="Metrics failed", 
            fps=fps, 
            detected=False, 
            landmarks={}, 
            body_angle=0.0, 
            hip_deviation=0.0
        )

    is_correct, feedback, hip_deviation = evaluate_plank_state(metrics)
    perf_time, imperf_time = _update_plank_session(session or "default", is_correct)
    fps = 1.0 / max(1e-6, time.time() - t0)

    pts = metrics["pts"]
    def obj(p):
        return {"x": float(p[0]), "y": float(p[1])}
    landmarks = {
        "l_shoulder": obj(pts["l_shoulder"]),
        "r_shoulder": obj(pts["r_shoulder"]),
        "l_hip": obj(pts["l_hip"]),
        "r_hip": obj(pts["r_hip"]),
        "l_wrist": obj(pts["l_wrist"]),
        "r_wrist": obj(pts["r_wrist"]),
        "l_ankle": obj(pts["l_ankle"]),
        "r_ankle": obj(pts["r_ankle"]),
        "nose": obj(pts["nose"]),
    }

    return PlankResponse(
        duration_perfect=perf_time,
        duration_imperfect=imperf_time,
        is_correct=is_correct,
        feedback=feedback,
        fps=fps,
        detected=True,
        landmarks=landmarks,
        body_angle=0.0, # Placeholder
        hip_deviation=float(hip_deviation)
    )


@app.websocket("/ws/plank")
async def ws_plank(websocket: WebSocket):
    await websocket.accept()
    print("[WS] /ws/plank connected from:", websocket.client)
    params = websocket.query_params
    flip_q = params.get("flip", "0")
    session_id = params.get("session", "default")
    flip_flag = 1 if str(flip_q) == "1" else 0

    try:
        while True:
            msg = await websocket.receive_json()
            t0 = time.time()
            b64 = msg.get("jpg_b64")
            if not b64:
                await websocket.send_json({"error": "missing jpg_b64"})
                continue

            try:
                img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                await websocket.send_json({"error": "decode_failed"})
                continue
            if img is None:
                await websocket.send_json({"error": "imdecode_failed"})
                continue

            h, w = img.shape[:2]
            max_w = 160
            if w > 0:
                scale = 160.0 / max(1.0, float(w))
                if scale < 1.0:
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
                    h, w = img.shape[:2]

            if flip_flag:
                img = cv2.flip(img, 1)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            lm = results.pose_landmarks.landmark if results.pose_landmarks else None
            
            if not lm:
                fps = 1.0 / max(1e-6, time.time() - t0)
                s = _plank_sessions.get(session_id, {"perfect_time": 0.0, "imperfect_time": 0.0})
                await websocket.send_json({
                    "duration_perfect": float(s["perfect_time"]),
                    "duration_imperfect": float(s["imperfect_time"]),
                    "is_correct": False,
                    "feedback": "No pose",
                    "fps": fps,
                    "detected": False,
                    "landmarks": {}
                })
                continue

            metrics = compute_metrics(lm, h, w)
            if metrics is None:
                continue

            is_correct, feedback, hip_deviation = evaluate_plank_state(metrics)
            perf_time, imperf_time = _update_plank_session(session_id, is_correct)
            fps = 1.0 / max(1e-6, time.time() - t0)

            pts = metrics["pts"]
            def obj(p):
                return {"x": float(p[0]), "y": float(p[1])}
            landmarks = {
                "l_shoulder": obj(pts["l_shoulder"]),
                "r_shoulder": obj(pts["r_shoulder"]),
                "l_hip": obj(pts["l_hip"]),
                "r_hip": obj(pts["r_hip"]),
                "l_wrist": obj(pts["l_wrist"]),
                "r_wrist": obj(pts["r_wrist"]),
                "l_ankle": obj(pts["l_ankle"]),
                "r_ankle": obj(pts["r_ankle"]),
                "nose": obj(pts["nose"]),
            }

            await websocket.send_json({
                "duration_perfect": perf_time,
                "duration_imperfect": imperf_time,
                "is_correct": is_correct,
                "feedback": feedback,
                "fps": fps,
                "detected": True,
                "landmarks": landmarks,
                "hip_deviation": float(hip_deviation)
            })

    except WebSocketDisconnect:
        print("[WS] /ws/plank disconnected:", websocket.client)
        return


class AnalyzePlankResponse(BaseModel):
    duration_total: float
    duration_perfect: float
    duration_imperfect: float
    processed_fps: float
    diagnostics: list | None = None


def _analyze_plank_video(path: str, flip: bool = False, sample_stride: int = 2):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0, 0.0, 0.0, 0.0, []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    video_duration = total_frames / fps if fps > 0 else 0.0
    
    # We will estimate time based on frame count and FPS
    frame_time = 1.0 / fps if fps > 0 else 0.033
    
    perfect_frames = 0
    imperfect_frames = 0
    processed = 0
    idx = 0
    diagnostics = []
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if sample_stride > 1 and (idx % sample_stride) != 0:
            idx += 1
            continue
        idx += 1
        
        if flip:
            frame = cv2.flip(frame, 1)
            
        h, w = frame.shape[:2]
        max_w = 160
        if w > max_w:
            scale = max_w / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            h, w = frame.shape[:2]
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        lm = results.pose_landmarks.landmark if results.pose_landmarks else None
        
        if lm:
            metrics = compute_metrics(lm, h, w)
            if metrics:
                is_correct, feedback, dev = evaluate_plank_state(metrics)
                if is_correct:
                    perfect_frames += 1
                else:
                    imperfect_frames += 1
                    # Log occasional diagnostics
                    if len(diagnostics) < 10 and processed % 30 == 0:
                        diagnostics.append({
                            "time": processed * frame_time * sample_stride,
                            "feedback": feedback,
                            "deviation": dev
                        })
        
        processed += 1
        
    cap.release()
    
    # Scale frames to time
    # We processed every 'sample_stride' frames
    # Each processed frame represents 'sample_stride' frames of time
    total_processed_time = processed * frame_time * sample_stride
    
    # Proportions
    if processed > 0:
        perf_time = (perfect_frames / processed) * total_processed_time
        imperf_time = (imperfect_frames / processed) * total_processed_time
    else:
        perf_time = 0.0
        imperf_time = 0.0
        
    proc_fps = processed / video_duration if video_duration > 0 else 0.0
    
    return video_duration, perf_time, imperf_time, proc_fps, diagnostics


@app.post("/analyze_plank", response_model=AnalyzePlankResponse)
async def analyze_plank(file: UploadFile = File(...), flip: Optional[int] = Form(default=0)):
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "video.mp4")[1] or ".mp4") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        dur, perf, imperf, p_fps, diag = _analyze_plank_video(tmp_path, flip=bool(flip), sample_stride=2)
        return AnalyzePlankResponse(
            duration_total=dur,
            duration_perfect=perf,
            duration_imperfect=imperf,
            processed_fps=p_fps,
            diagnostics=diag
        )
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


# ---------------------- High Knees (server-side) ----------------------
class HighKneesResponse(BaseModel):
    reps: int
    reps_clean: int
    reps_wrong: int
    phase: str
    feedback: str
    fps: float
    detected: bool
    landmarks: Dict[str, Dict[str, float]]


# Per-session state for High Knees
# Stores: {"reps": int, "reps_clean": int, "reps_wrong": int, "phase": str, "last_feedback": str}
_high_knees_sessions: Dict[str, Dict[str, int | str]] = {}


def evaluate_high_knees_state(metrics):
    """
    Evaluate High Knees state.
    State Machine: GROUND -> UP -> GROUND
    UP means at least one knee is significantly above the hip.
    """
    pts = metrics["pts"]
    
    # Y increases downwards
    l_hip_y = pts["l_hip"][1]
    r_hip_y = pts["r_hip"][1]
    l_knee_y = pts["l_knee"][1]
    r_knee_y = pts["r_knee"][1]
    l_shoulder_y = pts["l_shoulder"][1]
    r_shoulder_y = pts["r_shoulder"][1]
    
    # Average hip height
    hip_y = (l_hip_y + r_hip_y) / 2.0
    shoulder_y = (l_shoulder_y + r_shoulder_y) / 2.0
    
    # Check if either knee is high enough
    # "High" means knee Y is smaller (higher) than hip Y
    # Threshold: Knee should be at least at hip level or higher
    # Let's use a small buffer: knee < hip_y + 0.05 * shoulder_width (allow slightly below hip)
    # But for "clean" rep, knee < hip_y (strictly higher)
    
    shoulder_width = metrics["shoulder_width"]
    threshold = hip_y + 0.1 * shoulder_width # Permissive threshold for detection
    
    l_up = l_knee_y < threshold
    r_up = r_knee_y < threshold
    
    is_up = l_up or r_up
    
    # Quality check
    # Clean if knee is strictly above hip (smaller y)
    clean_threshold = hip_y
    l_clean = l_knee_y < clean_threshold
    r_clean = r_knee_y < clean_threshold
    is_clean = l_clean or r_clean
    
    # Detailed feedback
    feedback_parts = []
    
    if is_up and not is_clean:
        # Calculate how much higher the knee needs to go
        # Find which knee is being lifted (the higher one)
        active_knee_y = min(l_knee_y, r_knee_y)  # smaller y = higher
        knee_to_hip_diff = hip_y - active_knee_y  # positive if knee is below hip
        
        if knee_to_hip_diff > 0:
            feedback_parts.append("Lift knee higher - bring it above hip level")
        else:
            feedback_parts.append("Knee barely at hip level - lift higher for proper form")
    
    # Check for trunk lean (shoulders should stay relatively upright)
    # If shoulders move down significantly, person might be leaning forward
    # This is a simple heuristic
    trunk_lean_threshold = 0.15 * shoulder_width
    if is_up:
        # During high knees, shoulder shouldn't drop much
        # We can't compare to initial position easily, but we can check if torso is collapsing
        torso_length = abs(shoulder_y - hip_y)
        if torso_length < 0.5 * shoulder_width:  # Very compressed torso
            feedback_parts.append("Keep torso upright - avoid leaning forward")
    
    # Check alternation (both knees shouldn't be up at same time for high knees)
    if l_up and r_up and (l_knee_y < clean_threshold and r_knee_y < clean_threshold):
        feedback_parts.append("Alternate legs - only one knee up at a time")
    
    if not feedback_parts:
        feedback = "Good form!"
    else:
        feedback = " | ".join(feedback_parts)
        
    return "UP" if is_up else "GROUND", is_clean, feedback


def _update_high_knees_session(session_id: str, state: str, is_clean: bool, feedback: str) -> Tuple[int, int, int, str, str]:
    s = _high_knees_sessions.get(session_id, {
        "reps": 0, 
        "reps_clean": 0, 
        "reps_wrong": 0, 
        "phase": "GROUND",
        "last_feedback": ""
    })
    
    phase = str(s["phase"])
    reps = int(s["reps"])
    reps_clean = int(s["reps_clean"])
    reps_wrong = int(s["reps_wrong"])
    
    # State machine: GROUND -> UP -> GROUND (count rep on return to GROUND? or on entering UP?)
    # Usually count on completion (return to GROUND) or peak (UP).
    # Let's count on entering UP to give immediate feedback, but might double count if jittery.
    # Better: Count on UP -> GROUND transition? Or GROUND -> UP.
    # Let's try GROUND -> UP.
    
    if phase == "GROUND" and state == "UP":
        phase = "UP"
        reps += 1
        if is_clean:
            reps_clean += 1
        else:
            reps_wrong += 1
        s["last_feedback"] = feedback
        
    elif phase == "UP" and state == "GROUND":
        phase = "GROUND"
        
    s["phase"] = phase
    s["reps"] = reps
    s["reps_clean"] = reps_clean
    s["reps_wrong"] = reps_wrong
    
    _high_knees_sessions[session_id] = s
    
    return reps, reps_clean, reps_wrong, phase, str(s["last_feedback"])


@app.post("/high_knees", response_model=HighKneesResponse)
async def high_knees(
    file: UploadFile = File(...),
    flip: Optional[int] = Form(default=0),
    session: Optional[str] = Form(default="default"),
):
    t0 = time.time()
    data = await file.read()
    
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return HighKneesResponse(
            reps=0, reps_clean=0, reps_wrong=0, phase="GROUND", feedback="No image", 
            fps=0.0, detected=False, landmarks={}
        )

    h, w = img.shape[:2]
    max_w = 160
    if w > max_w:
        scale = max_w / float(w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]

    if flip:
        img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    lm = results.pose_landmarks.landmark if results.pose_landmarks else None
    
    if not lm:
        fps = 1.0 / max(1e-6, time.time() - t0)
        s = _high_knees_sessions.get(session or "default", {})
        return HighKneesResponse(
            reps=int(s.get("reps", 0)), 
            reps_clean=int(s.get("reps_clean", 0)), 
            reps_wrong=int(s.get("reps_wrong", 0)), 
            phase=str(s.get("phase", "GROUND")), 
            feedback="No pose", 
            fps=fps, 
            detected=False, 
            landmarks={}
        )

    metrics = compute_metrics(lm, h, w)
    if metrics is None:
        fps = 1.0 / max(1e-6, time.time() - t0)
        s = _high_knees_sessions.get(session or "default", {})
        return HighKneesResponse(
            reps=int(s.get("reps", 0)), 
            reps_clean=int(s.get("reps_clean", 0)), 
            reps_wrong=int(s.get("reps_wrong", 0)), 
            phase=str(s.get("phase", "GROUND")), 
            feedback="Metrics failed", 
            fps=fps, 
            detected=False, 
            landmarks={}
        )

    state, is_clean, feedback = evaluate_high_knees_state(metrics)
    reps, r_clean, r_wrong, phase, last_feedback = _update_high_knees_session(session or "default", state, is_clean, feedback)
    fps = 1.0 / max(1e-6, time.time() - t0)

    pts = metrics["pts"]
    def obj(p):
        return {"x": float(p[0]), "y": float(p[1])}
    landmarks = {
        "l_shoulder": obj(pts["l_shoulder"]),
        "r_shoulder": obj(pts["r_shoulder"]),
        "l_hip": obj(pts["l_hip"]),
        "r_hip": obj(pts["r_hip"]),
        "l_knee": obj(pts["l_knee"]),
        "r_knee": obj(pts["r_knee"]),
        "l_ankle": obj(pts["l_ankle"]),
        "r_ankle": obj(pts["r_ankle"]),
        "nose": obj(pts["nose"]),
    }

    return HighKneesResponse(
        reps=reps,
        reps_clean=r_clean,
        reps_wrong=r_wrong,
        phase=phase,
        feedback=last_feedback,
        fps=fps,
        detected=True,
        landmarks=landmarks
    )


@app.websocket("/ws/high_knees")
async def ws_high_knees(websocket: WebSocket):
    await websocket.accept()
    print("[WS] /ws/high_knees connected from:", websocket.client)
    params = websocket.query_params
    flip_q = params.get("flip", "0")
    session_id = params.get("session", "default")
    flip_flag = 1 if str(flip_q) == "1" else 0

    try:
        while True:
            msg = await websocket.receive_json()
            t0 = time.time()
            b64 = msg.get("jpg_b64")
            if not b64:
                await websocket.send_json({"error": "missing jpg_b64"})
                continue

            try:
                img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                await websocket.send_json({"error": "decode_failed"})
                continue
            if img is None:
                await websocket.send_json({"error": "imdecode_failed"})
                continue

            h, w = img.shape[:2]
            max_w = 160
            if w > 0:
                scale = 160.0 / max(1.0, float(w))
                if scale < 1.0:
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
                    h, w = img.shape[:2]

            if flip_flag:
                img = cv2.flip(img, 1)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            lm = results.pose_landmarks.landmark if results.pose_landmarks else None
            
            if not lm:
                fps = 1.0 / max(1e-6, time.time() - t0)
                s = _high_knees_sessions.get(session_id, {})
                await websocket.send_json({
                    "reps": int(s.get("reps", 0)),
                    "reps_clean": int(s.get("reps_clean", 0)),
                    "reps_wrong": int(s.get("reps_wrong", 0)),
                    "phase": str(s.get("phase", "GROUND")),
                    "feedback": "No pose",
                    "fps": fps,
                    "detected": False,
                    "landmarks": {}
                })
                continue

            metrics = compute_metrics(lm, h, w)
            if metrics is None:
                continue

            state, is_clean, feedback = evaluate_high_knees_state(metrics)
            reps, r_clean, r_wrong, phase, last_feedback = _update_high_knees_session(session_id, state, is_clean, feedback)
            fps = 1.0 / max(1e-6, time.time() - t0)

            pts = metrics["pts"]
            def obj(p):
                return {"x": float(p[0]), "y": float(p[1])}
            landmarks = {
                "l_shoulder": obj(pts["l_shoulder"]),
                "r_shoulder": obj(pts["r_shoulder"]),
                "l_hip": obj(pts["l_hip"]),
                "r_hip": obj(pts["r_hip"]),
                "l_knee": obj(pts["l_knee"]),
                "r_knee": obj(pts["r_knee"]),
                "l_ankle": obj(pts["l_ankle"]),
                "r_ankle": obj(pts["r_ankle"]),
                "nose": obj(pts["nose"]),
            }

            await websocket.send_json({
                "reps": reps,
                "reps_clean": r_clean,
                "reps_wrong": r_wrong,
                "phase": phase,
                "feedback": last_feedback,
                "fps": fps,
                "detected": True,
                "landmarks": landmarks
            })

    except WebSocketDisconnect:
        print("[WS] /ws/high_knees disconnected:", websocket.client)
        return


class AnalyzeHighKneesResponse(BaseModel):
    reps: int
    reps_clean: int
    reps_wrong: int
    processed_fps: float
    diagnostics: list | None = None


def _analyze_high_knees_video(path: str, flip: bool = False, sample_stride: int = 2):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0, 0, 0, 0.0, []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    video_duration = total_frames / fps if fps > 0 else 0.0
    frame_time = 1.0 / fps if fps > 0 else 0.033
    
    reps = 0
    reps_clean = 0
    reps_wrong = 0
    phase = "GROUND"
    processed = 0
    idx = 0
    diagnostics = []
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if sample_stride > 1 and (idx % sample_stride) != 0:
            idx += 1
            continue
        idx += 1
        
        if flip:
            frame = cv2.flip(frame, 1)
            
        h, w = frame.shape[:2]
        max_w = 160
        if w > max_w:
            scale = max_w / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            h, w = frame.shape[:2]
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        lm = results.pose_landmarks.landmark if results.pose_landmarks else None
        
        if lm:
            metrics = compute_metrics(lm, h, w)
            if metrics:
                state, is_clean, feedback = evaluate_high_knees_state(metrics)
                
                if phase == "GROUND" and state == "UP":
                    phase = "UP"
                    reps += 1
                    # Track ALL reps with feedback
                    diagnostics.append({
                        "time": processed * frame_time * sample_stride,
                        "rep": reps,
                        "feedback": feedback,
                        "is_clean": is_clean
                    })
                    if is_clean:
                        reps_clean += 1
                    else:
                        reps_wrong += 1
                elif phase == "UP" and state == "GROUND":
                    phase = "GROUND"
        
        processed += 1
        
    cap.release()
    proc_fps = processed / video_duration if video_duration > 0 else 0.0
    
    return reps, reps_clean, reps_wrong, proc_fps, diagnostics


@app.post("/analyze_high_knees", response_model=AnalyzeHighKneesResponse)
async def analyze_high_knees(file: UploadFile = File(...), flip: Optional[int] = Form(default=0)):
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "video.mp4")[1] or ".mp4") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        r, rc, rw, p_fps, diag = _analyze_high_knees_video(tmp_path, flip=bool(flip), sample_stride=2)
        return AnalyzeHighKneesResponse(
            reps=r,
            reps_clean=rc,
            reps_wrong=rw,
            processed_fps=p_fps,
            diagnostics=diag
        )
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


# ---------------------- Mountain Climbers (server-side) ----------------------
class MountainClimbersResponse(BaseModel):
    reps: int
    reps_clean: int
    reps_wrong: int
    phase: str
    feedback: str
    fps: float
    detected: bool
    landmarks: Dict[str, Dict[str, float]]


# Per-session state for Mountain Climbers
_mountain_climbers_sessions: Dict[str, Dict[str, int | str]] = {}


def evaluate_mountain_climbers_state(metrics):
    """
    Evaluate Mountain Climbers state with comprehensive form analysis.
    Checks for proper plank position and knee drive.
    Works for both side and front views.
    Each knee drive counts as 1 rep.
    
    Correct Form:
    - Body in straight line (head → hips → heels)
    - Hands under shoulders
    - Hips stable and level
    - Knee drives toward chest
    - Core tight, minimal torso rotation
    """
    pts = metrics["pts"]
    
    # Get key points
    l_shoulder_x = pts["l_shoulder"][0]
    r_shoulder_x = pts["r_shoulder"][0]
    l_shoulder_y = pts["l_shoulder"][1]
    r_shoulder_y = pts["r_shoulder"][1]
    l_hip_y = pts["l_hip"][1]
    r_hip_y = pts["r_hip"][1]
    l_knee_y = pts["l_knee"][1]
    r_knee_y = pts["r_knee"][1]
    l_ankle_y = pts["l_ankle"][1]
    r_ankle_y = pts["r_ankle"][1]
    
    shoulder_x = (l_shoulder_x + r_shoulder_x) / 2.0
    shoulder_y = (l_shoulder_y + r_shoulder_y) / 2.0
    hip_y = (l_hip_y + r_hip_y) / 2.0
    ankle_y = (l_ankle_y + r_ankle_y) / 2.0
    
    shoulder_width = metrics["shoulder_width"]
    
    # 1. Check plank position (body should be straight: head → hips → heels)
    shoulder_ankle_mid_y = (shoulder_y + ankle_y) / 2.0
    hip_deviation = abs(hip_y - shoulder_ankle_mid_y) / shoulder_width
    
    # Categorize plank quality
    good_plank = hip_deviation < 0.8
    
    # 2. Detect knee drive
    baseline_knee_y = ankle_y
    l_knee_raised = (baseline_knee_y - l_knee_y) > 0.15 * shoulder_width
    r_knee_raised = (baseline_knee_y - r_knee_y) > 0.15 * shoulder_width
    
    # Determine which leg is up
    active_leg = None
    if l_knee_raised and not r_knee_raised:
        active_leg = "LEFT"
    elif r_knee_raised and not l_knee_raised:
        active_leg = "RIGHT"
    elif l_knee_raised and r_knee_raised:
        # Both up - pick the one that's higher
        if l_knee_y < r_knee_y:
            active_leg = "LEFT"
        else:
            active_leg = "RIGHT"
    
    is_knee_up = l_knee_raised or r_knee_raised
    
    # Detailed quality checks based on form guidelines
    feedback_parts = []
    
    # Check 1: Hips too high (pike position)
    if hip_y < shoulder_ankle_mid_y - 0.3 * shoulder_width:
        feedback_parts.append("Lower your hips - keep body in straight line")
    
    # Check 2: Hips sagging (lower back dipping)
    elif hip_y > shoulder_ankle_mid_y + 0.3 * shoulder_width:
        feedback_parts.append("Engage core - don't let hips sag")
    
    # Check 3: Knee drive quality (should reach toward chest)
    if is_knee_up:
        active_knee_y = l_knee_y if (active_leg == "LEFT") else r_knee_y
        
        # Knee should come significantly higher than hip
        # Good form: knee reaches toward chest (close to shoulder level)
        knee_to_shoulder_dist = active_knee_y - shoulder_y
        
        if knee_to_shoulder_dist > 0.4 * shoulder_width:
            # Knee is not high enough - short drive
            feedback_parts.append("Drive knee closer to chest for full range")
        elif knee_to_shoulder_dist > 0.2 * shoulder_width:
            # Moderate drive
            feedback_parts.append("Bring knee higher toward chest")
    
    # Determine if this is a clean rep
    # Clean rep criteria:
    # 1. Good plank position (body straight)
    # 2. Knee is raised
    # 3. Knee reaches reasonably toward chest
    if is_knee_up:
        active_knee_y = l_knee_y if (active_leg == "LEFT") else r_knee_y
        knee_height_good = (active_knee_y < hip_y)  # Knee above hip level
        is_clean = good_plank and knee_height_good
    else:
        is_clean = False
    
    # Generate feedback message
    if not feedback_parts and is_knee_up:
        feedback = "Good form!"
    elif not feedback_parts:
        feedback = "Maintain plank position"
    else:
        feedback = " | ".join(feedback_parts)
    
    return active_leg, is_clean, feedback


def _update_mountain_climbers_session(session_id: str, active_leg: str | None, is_clean: bool, feedback: str) -> Tuple[int, int, int, str, str]:
    s = _mountain_climbers_sessions.get(session_id, {
        "reps": 0, 
        "reps_clean": 0, 
        "reps_wrong": 0, 
        "last_leg": None,
        "last_feedback": ""
    })
    
    last_leg = s.get("last_leg")
    reps = int(s["reps"])
    reps_clean = int(s["reps_clean"])
    reps_wrong = int(s["reps_wrong"])
    
    # Count a rep when a leg is raised and it's different from the last leg
    # OR if no leg was previously raised (first rep)
    if active_leg is not None and active_leg != last_leg:
        reps += 1
        if is_clean:
            reps_clean += 1
        else:
            reps_wrong += 1
        s["last_feedback"] = feedback
        s["last_leg"] = active_leg
    elif active_leg is None:
        # Reset last_leg when back in plank
        s["last_leg"] = None
    
    s["reps"] = reps
    s["reps_clean"] = reps_clean
    s["reps_wrong"] = reps_wrong
    
    _mountain_climbers_sessions[session_id] = s
    
    phase = "KNEE_UP" if active_leg else "PLANK"
    return reps, reps_clean, reps_wrong, phase, str(s.get("last_feedback", ""))


@app.post("/mountain_climbers", response_model=MountainClimbersResponse)
async def mountain_climbers(
    file: UploadFile = File(...),
    flip: Optional[int] = Form(default=0),
    session: Optional[str] = Form(default="default"),
):
    t0 = time.time()
    data = await file.read()
    
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return MountainClimbersResponse(
            reps=0, reps_clean=0, reps_wrong=0, phase="PLANK", feedback="No image", 
            fps=0.0, detected=False, landmarks={}
        )

    h, w = img.shape[:2]
    max_w = 160
    if w > max_w:
        scale = max_w / float(w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]

    if flip:
        img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    lm = results.pose_landmarks.landmark if results.pose_landmarks else None
    
    if not lm:
        fps = 1.0 / max(1e-6, time.time() - t0)
        s = _mountain_climbers_sessions.get(session or "default", {})
        return MountainClimbersResponse(
            reps=int(s.get("reps", 0)), 
            reps_clean=int(s.get("reps_clean", 0)), 
            reps_wrong=int(s.get("reps_wrong", 0)), 
            phase=str(s.get("phase", "PLANK")), 
            feedback="No pose", 
            fps=fps, 
            detected=False, 
            landmarks={}
        )

    metrics = compute_metrics(lm, h, w)
    if metrics is None:
        fps = 1.0 / max(1e-6, time.time() - t0)
        s = _mountain_climbers_sessions.get(session or "default", {})
        return MountainClimbersResponse(
            reps=int(s.get("reps", 0)), 
            reps_clean=int(s.get("reps_clean", 0)), 
            reps_wrong=int(s.get("reps_wrong", 0)), 
            phase=str(s.get("phase", "PLANK")), 
            feedback="Metrics failed", 
            fps=fps, 
            detected=False, 
            landmarks={}
        )

    active_leg, is_clean, feedback = evaluate_mountain_climbers_state(metrics)
    reps, r_clean, r_wrong, phase, last_feedback = _update_mountain_climbers_session(session or "default", active_leg, is_clean, feedback)
    fps = 1.0 / max(1e-6, time.time() - t0)

    pts = metrics["pts"]
    def obj(p):
        return {"x": float(p[0]), "y": float(p[1])}
    landmarks = {
        "l_shoulder": obj(pts["l_shoulder"]),
        "r_shoulder": obj(pts["r_shoulder"]),
        "l_hip": obj(pts["l_hip"]),
        "r_hip": obj(pts["r_hip"]),
        "l_knee": obj(pts["l_knee"]),
        "r_knee": obj(pts["r_knee"]),
        "l_ankle": obj(pts["l_ankle"]),
        "r_ankle": obj(pts["r_ankle"]),
        "nose": obj(pts["nose"]),
    }

    return MountainClimbersResponse(
        reps=reps,
        reps_clean=r_clean,
        reps_wrong=r_wrong,
        phase=phase,
        feedback=last_feedback,
        fps=fps,
        detected=True,
        landmarks=landmarks
    )


@app.websocket("/ws/mountain_climbers")
async def ws_mountain_climbers(websocket: WebSocket):
    await websocket.accept()
    print("[WS] /ws/mountain_climbers connected from:", websocket.client)
    params = websocket.query_params
    flip_q = params.get("flip", "0")
    session_id = params.get("session", "default")
    flip_flag = 1 if str(flip_q) == "1" else 0

    try:
        while True:
            msg = await websocket.receive_json()
            t0 = time.time()
            b64 = msg.get("jpg_b64")
            if not b64:
                await websocket.send_json({"error": "missing jpg_b64"})
                continue

            try:
                img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                await websocket.send_json({"error": "decode_failed"})
                continue
            if img is None:
                await websocket.send_json({"error": "imdecode_failed"})
                continue

            h, w = img.shape[:2]
            max_w = 160
            if w > 0:
                scale = 160.0 / max(1.0, float(w))
                if scale < 1.0:
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
                    h, w = img.shape[:2]

            if flip_flag:
                img = cv2.flip(img, 1)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            lm = results.pose_landmarks.landmark if results.pose_landmarks else None
            
            if not lm:
                fps = 1.0 / max(1e-6, time.time() - t0)
                s = _mountain_climbers_sessions.get(session_id, {})
                await websocket.send_json({
                    "reps": int(s.get("reps", 0)),
                    "reps_clean": int(s.get("reps_clean", 0)),
                    "reps_wrong": int(s.get("reps_wrong", 0)),
                    "phase": str(s.get("phase", "PLANK")),
                    "feedback": "No pose",
                    "fps": fps,
                    "detected": False,
                    "landmarks": {}
                })
                continue

            metrics = compute_metrics(lm, h, w)
            if metrics is None:
                continue

            active_leg, is_clean, feedback = evaluate_mountain_climbers_state(metrics)
            reps, r_clean, r_wrong, phase, last_feedback = _update_mountain_climbers_session(session_id, active_leg, is_clean, feedback)
            fps = 1.0 / max(1e-6, time.time() - t0)

            pts = metrics["pts"]
            def obj(p):
                return {"x": float(p[0]), "y": float(p[1])}
            landmarks = {
                "l_shoulder": obj(pts["l_shoulder"]),
                "r_shoulder": obj(pts["r_shoulder"]),
                "l_hip": obj(pts["l_hip"]),
                "r_hip": obj(pts["r_hip"]),
                "l_knee": obj(pts["l_knee"]),
                "r_knee": obj(pts["r_knee"]),
                "l_ankle": obj(pts["l_ankle"]),
                "r_ankle": obj(pts["r_ankle"]),
                "nose": obj(pts["nose"]),
            }

            await websocket.send_json({
                "reps": reps,
                "reps_clean": r_clean,
                "reps_wrong": r_wrong,
                "phase": phase,
                "feedback": last_feedback,
                "fps": fps,
                "detected": True,
                "landmarks": landmarks
            })

    except WebSocketDisconnect:
        print("[WS] /ws/mountain_climbers disconnected:", websocket.client)
        return


class AnalyzeMountainClimbersResponse(BaseModel):
    reps: int
    reps_clean: int
    reps_wrong: int
    processed_fps: float
    diagnostics: list | None = None


def _analyze_mountain_climbers_video(path: str, flip: bool = False, sample_stride: int = 2):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0, 0, 0, 0.0, []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    video_duration = total_frames / fps if fps > 0 else 0.0
    frame_time = 1.0 / fps if fps > 0 else 0.033
    
    reps = 0
    reps_clean = 0
    reps_wrong = 0
    last_leg = None
    processed = 0
    idx = 0
    diagnostics = []
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if sample_stride > 1 and (idx % sample_stride) != 0:
            idx += 1
            continue
        idx += 1
        
        if flip:
            frame = cv2.flip(frame, 1)
            
        h, w = frame.shape[:2]
        max_w = 160
        if w > max_w:
            scale = max_w / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            h, w = frame.shape[:2]
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        lm = results.pose_landmarks.landmark if results.pose_landmarks else None
        
        if lm:
            metrics = compute_metrics(lm, h, w)
            if metrics:
                active_leg, is_clean, feedback = evaluate_mountain_climbers_state(metrics)
                
                # Track which leg was last raised to count each knee drive
                if active_leg is not None and active_leg != last_leg:
                    reps += 1
                    last_leg = active_leg
                    # Track ALL reps with feedback
                    diagnostics.append({
                        "time": processed * frame_time * sample_stride,
                        "rep": reps,
                        "feedback": feedback,
                        "is_clean": bool(is_clean)
                    })
                    if is_clean:
                        reps_clean += 1
                    else:
                        reps_wrong += 1
                elif active_leg is None:
                    # Reset when back in plank
                    last_leg = None
        
        processed += 1
        
    cap.release()
    proc_fps = processed / video_duration if video_duration > 0 else 0.0
    
    return reps, reps_clean, reps_wrong, proc_fps, diagnostics


@app.post("/analyze_mountain_climbers", response_model=AnalyzeMountainClimbersResponse)
async def analyze_mountain_climbers(file: UploadFile = File(...), flip: Optional[int] = Form(default=0)):
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "video.mp4")[1] or ".mp4") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        r, rc, rw, p_fps, diag = _analyze_mountain_climbers_video(tmp_path, flip=bool(flip), sample_stride=2)
        return AnalyzeMountainClimbersResponse(
            reps=r,
            reps_clean=rc,
            reps_wrong=rw,
            processed_fps=p_fps,
            diagnostics=diag
        )
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
