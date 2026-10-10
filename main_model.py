import os
import time
import threading
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# ---------- SETTINGS (change these to tune speed) ----------
MODEL_NAME = "yolov8s.pt"   # n = fastest, s = more accurate but slower
IMG_SIZE = 640              # smaller = faster (try 480, 640, 736)
SKIP_FRAMES = 3             # detect on every 3rd frame (1 = every frame)
CONF = 0.25
IOU = 0.50
JPEG_QUALITY = 75           # stream quality (lower = faster)
FRAME_SIZE = (960, 540)     # (width, height) every frame is resized to
# default thresholds per mode: (safe_max, warning_max)
THRESHOLDS = {"live": (3, 8), "video": (18, 40)}
# -----------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXT = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB upload limit

# ---------------------------------------------------------------------------
# Model (loaded once, warmed up so the first real frame is not slow)
# ---------------------------------------------------------------------------
print("Loading model:", MODEL_NAME)
model = YOLO(MODEL_NAME)
model(np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), np.uint8),
      imgsz=IMG_SIZE, verbose=False)
print("Model ready.")


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
state_lock = threading.Lock()
state = {
    "running": False,
    "mode": None,            # "live" or "video"
    "source": None,          # file name or "Webcam"
    "loop": True,
    "people": 0,
    "bicycle": 0,
    "car": 0,
    "bike": 0,               # motorcycles
    "peak": 0,
    "status": "IDLE",        # IDLE / SAFE / WARNING / CAUTION
    "advice": "Choose a source to start monitoring",
    "safe_max": THRESHOLDS["live"][0],
    "warning_max": THRESHOLDS["live"][1],
    "fps": 0.0,
    "infer_ms": 0,
    "message": "",
    "error": None,
}
history = deque(maxlen=120)   # one point per second
alerts = deque(maxlen=50)     # status changes, newest first

# latest JPEG frame shared with the /video_feed generators
frame_cond = threading.Condition()
latest = {"jpeg": None, "id": 0}

# worker control
ctl_lock = threading.Lock()
worker_thread = None
stop_event = None


def make_text_frame(title, sub=""):
    """A plain dark frame with centred text (idle / error screen)."""
    img = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), (26, 20, 16), np.uint8)
    lines = ((title, 1.1, (235, 235, 235)), (sub, 0.6, (160, 160, 160)))
    for i, (text, scale, col) in enumerate(lines):
        if not text:
            continue
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, scale, 2)
        x = (FRAME_SIZE[0] - w) // 2
        y = FRAME_SIZE[1] // 2 + (0 if i == 0 else 45)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, scale, col, 2)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def publish(jpeg):
    with frame_cond:
        latest["jpeg"] = jpeg
        latest["id"] += 1
        frame_cond.notify_all()


IDLE_JPEG = make_text_frame("CrowdGuard AI",
                            "Choose a source below to start monitoring")
publish(IDLE_JPEG)


def classify(people, safe_max, warning_max):
    """Same rules as the original script. Returns (status, advice, BGR colour)."""
    if people < safe_max:
        return "SAFE", "Enjoy", (0, 255, 0)
    elif people <= warning_max:
        return "WARNING", "Please don't run", (0, 165, 255)
    return "CAUTION", "EVACUATE IMMEDIATELY", (0, 0, 255)


# ---------------------------------------------------------------------------
# Detection worker (runs in a background thread)
# ---------------------------------------------------------------------------
def detection_worker(mode, path, loop, stop):
    cap = cv2.VideoCapture(0) if mode == "live" else cv2.VideoCapture(path)
    if mode == "live":
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        err = ("Could not open the camera. Is it used by another app?"
               if mode == "live" else "Could not open this video file.")
        with state_lock:
            state.update(running=False, status="IDLE", advice="", error=err)
        publish(make_text_frame("Could not open source", err[:60]))
        cap.release()
        return

    # for video files, play at the natural speed instead of as fast as possible
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if not (1 <= src_fps <= 120):
        src_fps = 30.0
    frame_delay = 1.0 / src_fps if mode == "video" else 0.0

    frame_no = 0
    smooth_count = 0.0
    first_frame = True
    boxes = []                       # last detected boxes, reused on skipped frames
    person_count = bicycle_count = car_count = motorcycle_count = 0
    peak = 0
    last_status = None
    last_hist = 0.0
    fps_ema = 0.0
    infer_ms = 0.0
    read_since_reset = 0

    try:
        while not stop.is_set():
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                # loop the video, but never spin forever on an unreadable file
                if mode == "video" and loop and read_since_reset > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    read_since_reset = 0
                    continue
                with state_lock:
                    state.update(running=False, message="Video finished")
                break
            read_since_reset += 1

            # make the frame smaller first (faster for everything)
            frame = cv2.resize(frame, FRAME_SIZE)

            # run detection only on some frames
            if frame_no % SKIP_FRAMES == 0:
                ts = time.time()
                results = model(frame, classes=[0, 1, 2, 3], conf=CONF, iou=IOU,
                                imgsz=IMG_SIZE, verbose=False)
                dt = (time.time() - ts) * 1000
                infer_ms = dt if infer_ms == 0 else 0.8 * infer_ms + 0.2 * dt

                boxes = results[0].boxes.xyxy.cpu().numpy()
                class_ids = [int(c) for c in results[0].boxes.cls.cpu().tolist()]

                person_count = class_ids.count(0)
                bicycle_count = class_ids.count(1)
                car_count = class_ids.count(2)
                motorcycle_count = class_ids.count(3)

                # smooth the count so it does not jump around
                if first_frame:
                    smooth_count = person_count
                    first_frame = False
                else:
                    smooth_count = 0.6 * smooth_count + 0.4 * person_count

            frame_no += 1
            people = round(smooth_count)
            peak = max(peak, people)

            with state_lock:
                safe_max = state["safe_max"]
                warning_max = state["warning_max"]
            status, advice, color = classify(people, safe_max, warning_max)

            # draw boxes
            for box in boxes:
                x1, y1, x2, y2 = (int(v) for v in box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # small info bar on the video itself
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (FRAME_SIZE[0], 40), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            cv2.putText(frame, f"People: {people}    Status: {status}", (16, 27),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)

            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                publish(buf.tobytes())

            # keep video files at their natural speed
            elapsed = time.time() - t0
            if frame_delay > elapsed:
                stop.wait(frame_delay - elapsed)

            loop_dt = max(time.time() - t0, 1e-6)
            fps_ema = (1.0 / loop_dt) if fps_ema == 0 else 0.9 * fps_ema + 0.1 / loop_dt

            now = time.time()
            with state_lock:
                state.update(people=people, bicycle=bicycle_count, car=car_count,
                             bike=motorcycle_count, peak=peak, status=status,
                             advice=advice, fps=round(fps_ema, 1),
                             infer_ms=round(infer_ms))
                if status != last_status:
                    alerts.appendleft({"time": time.strftime("%H:%M:%S"),
                                       "status": status, "people": people})
                    last_status = status
                if now - last_hist >= 1.0:
                    history.append({"t": time.strftime("%H:%M:%S"), "people": people})
                    last_hist = now
    finally:
        cap.release()


def stop_worker():
    """Stop the running worker (call while holding ctl_lock)."""
    global worker_thread, stop_event
    if stop_event is not None:
        stop_event.set()
    if worker_thread is not None and worker_thread.is_alive():
        worker_thread.join(timeout=10)
    worker_thread = None
    stop_event = None


def start_worker(mode, path, source_name, loop):
    global worker_thread, stop_event
    with ctl_lock:
        stop_worker()
        safe_max, warning_max = THRESHOLDS[mode]
        with state_lock:
            state.update(running=True, mode=mode, source=source_name, loop=loop,
                         people=0, bicycle=0, car=0, bike=0, peak=0,
                         status="IDLE", advice="Starting...", fps=0.0,
                         infer_ms=0, message="", error=None,
                         safe_max=safe_max, warning_max=warning_max)
            history.clear()
            alerts.clear()
        stop_event = threading.Event()
        worker_thread = threading.Thread(
            target=detection_worker, args=(mode, path, loop, stop_event),
            daemon=True)
        worker_thread.start()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


def mjpeg_frames():
    last_id = -1
    while True:
        with frame_cond:
            # wake when a new frame arrives; after 1s resend the last one so a
            # closed browser tab is noticed and this generator can exit
            frame_cond.wait_for(lambda: latest["id"] != last_id, timeout=1.0)
            jpeg = latest["jpeg"]
            last_id = latest["id"]
        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
               + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")


@app.route("/video_feed")
def video_feed():
    resp = Response(mjpeg_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/stats")
def stats():
    with state_lock:
        payload = dict(state)
        payload["history"] = list(history)
        payload["alerts"] = list(alerts)[:12]
    return jsonify(payload)


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if f is None or not f.filename:
        return jsonify(ok=False, error="No file selected."), 400

    name = f.filename
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXT:
        return jsonify(ok=False, error="Unsupported file type. Use: "
                       + ", ".join(sorted(ALLOWED_EXT)) + "."), 400

    stem = secure_filename(name.rsplit(".", 1)[0]) or "video"
    final = f"{int(time.time())}_{stem}.{ext}"   # unique name, never overwrites
    f.save(os.path.join(UPLOAD_DIR, final))
    return jsonify(ok=True, filename=final)


@app.route("/videos")
def videos():
    items = []
    for fn in os.listdir(UPLOAD_DIR):
        if fn.rsplit(".", 1)[-1].lower() in ALLOWED_EXT:
            full = os.path.join(UPLOAD_DIR, fn)
            label = fn.split("_", 1)[1] if "_" in fn else fn
            items.append({"name": fn, "label": label,
                          "size_mb": round(os.path.getsize(full) / 1e6, 1),
                          "mtime": os.path.getmtime(full)})
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return jsonify(items)


@app.route("/start", methods=["POST"])
def start():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode")
    if mode == "live":
        start_worker("live", None, "Webcam", loop=False)
    elif mode == "video":
        name = os.path.basename(data.get("filename") or "")
        path = os.path.join(UPLOAD_DIR, name)
        if not name or not os.path.isfile(path):
            return jsonify(ok=False, error="Video not found. Upload one first."), 400
        start_worker("video", path, name.split("_", 1)[-1], bool(data.get("loop", True)))
    else:
        return jsonify(ok=False, error="mode must be 'live' or 'video'."), 400
    return jsonify(ok=True)


@app.route("/stop", methods=["POST"])
def stop():
    with ctl_lock:
        stop_worker()
        with state_lock:
            state.update(running=False, people=0, bicycle=0, car=0, bike=0,
                         status="IDLE", advice="Stopped", fps=0.0, infer_ms=0,
                         message="")
    publish(IDLE_JPEG)
    return jsonify(ok=True)


@app.route("/set_thresholds", methods=["POST"])
def set_thresholds():
    data = request.get_json(silent=True) or {}
    try:
        safe_max = int(data["safe_max"])
        warning_max = int(data["warning_max"])
    except (KeyError, TypeError, ValueError):
        return jsonify(ok=False, error="safe_max and warning_max must be numbers."), 400
    if not (1 <= safe_max <= warning_max <= 500):
        return jsonify(ok=False, error="Need 1 <= safe_max <= warning_max <= 500."), 400
    with state_lock:
        state.update(safe_max=safe_max, warning_max=warning_max)
    return jsonify(ok=True)


@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return jsonify(ok=False, error="File too large (limit is 1 GB)."), 413


if __name__ == "__main__":
    print("\n  CrowdGuard AI is running ->  http://127.0.0.1:5000\n")
    # use host="0.0.0.0" to also open the dashboard from a phone on the same WiFi
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
