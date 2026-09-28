"""
Drowsiness Detection System
============================
Monitors the user via webcam while watching a video.
When drowsiness/sleep is detected:
  - Closes the video application (VLC or any player)
  - Prints a message to the terminal
  - Logs the event to logs/drowsiness.log

Architecture:
  MediaPipe FaceLandmarker (Tasks API, v1.0+)
    → EAR (Eye Aspect Ratio) + PERCLOS
    → Drowsiness Decision
    → Kill video player + log

Requirements:
  mediapipe>=1.0.1, opencv-python>=5.0, numpy>=2.0
  Model file: models/face_landmarker.task
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import time
import logging
import sys
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_PATH = Path("models/face_landmarker.task")

# Thresholds
EAR_CLOSED_THRESHOLD       = 0.20   # EAR below this = eye closed
PERCLOS_WINDOW_FRAMES      = 60     # evaluate over last 60 frames (~2s at 30fps)
PERCLOS_DROWSY_THRESHOLD   = 0.40   # 40% closed frames = drowsy
CONSECUTIVE_CLOSED_SECONDS = 2.5   # hard rule: eyes closed 2.5s = asleep

WEBCAM_INDEX = 0

# Log file
LOG_DIR  = Path("logs")
LOG_FILE = LOG_DIR / "drowsiness.log"

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("DrowsinessDetector")


# ─────────────────────────────────────────────
# EAR COMPUTATION
# ─────────────────────────────────────────────
#
# MediaPipe FaceLandmarker returns 478 landmarks (with irises).
# Eye landmark indices (same as classic Face Mesh):
#
#   Left eye  (from subject's perspective):
#     p1=362, p2=385, p3=387, p4=263, p5=373, p6=380
#   Right eye:
#     p1=33,  p2=160, p3=158, p4=133, p5=153, p6=144
#
#         p2   p3
#   p1  ──────────  p4
#         p6   p5
#
#         ||p2-p6|| + ||p3-p5||
# EAR = ─────────────────────────
#              2 × ||p1-p4||

LEFT_EYE_INDICES  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33,  160, 158, 133, 153, 144]


def _pt(lm, img_w, img_h):
    """Convert a NormalizedLandmark to pixel coordinates."""
    return np.array([lm.x * img_w, lm.y * img_h])


def compute_ear(landmarks, eye_indices, img_w, img_h):
    """Compute Eye Aspect Ratio for one eye."""
    pts = [_pt(landmarks[i], img_w, img_h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts

    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h  = np.linalg.norm(p1 - p4)

    return (v1 + v2) / (2.0 * h + 1e-6)


# ─────────────────────────────────────────────
# VIDEO PLAYER SHUTDOWN
# ─────────────────────────────────────────────

VIDEO_PLAYER_PROCS = [
    "vlc.exe", "vlc",
    "mpv.exe", "mpv",
    "mpc-hc64.exe", "mpc-hc.exe",
    "wmplayer.exe",
    "Movies.UI.exe",
    "PotPlayerMini64.exe", "PotPlayerMini.exe",
]


def find_and_kill_video_players():
    """Kill any running video player. Returns list of killed process names."""
    killed = []
    try:
        result = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5
        )
        running = result.stdout.lower()
        for player in VIDEO_PLAYER_PROCS:
            if player.lower() in running:
                subprocess.run(
                    ["taskkill", "/F", "/IM", player],
                    capture_output=True, timeout=5
                )
                killed.append(player)
    except Exception as e:
        logger.error(f"Error killing video players: {e}")
    return killed


# ─────────────────────────────────────────────
# OVERLAY DRAWING
# ─────────────────────────────────────────────

def draw_overlay(frame, ear_l, ear_r, perclos, closed_duration, state):
    h, w = frame.shape[:2]
    avg_ear = (ear_l + ear_r) / 2.0

    # Semi-transparent background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (340, 175), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    color = {
        "ALERT":    (0, 220, 0),
        "DROWSY":   (0, 165, 255),
        "SLEEPING": (0, 0, 255),
        "NO FACE":  (180, 180, 180),
    }.get(state, (255, 255, 255))

    lines = [
        f"State    : {state}",
        f"EAR      : {avg_ear:.3f}  (L:{ear_l:.2f} R:{ear_r:.2f})",
        f"PERCLOS  : {perclos*100:.1f}%",
        f"Eyes shut: {closed_duration:.1f}s  (limit: {CONSECUTIVE_CLOSED_SECONDS}s)",
    ]
    y = 28
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, color, 1, cv2.LINE_AA)
        y += 36

    if state in ("DROWSY", "SLEEPING"):
        cv2.putText(frame, f"!!! {state} !!!",
                    (w // 2 - 100, h - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────
# SLEEP EVENT HANDLER
# ─────────────────────────────────────────────

def _trigger_shutdown(closed_duration: float, perclos: float):
    """Called once when sleep is confirmed. Prints, logs, kills player."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 60

    # ── Terminal ──────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  DROWSINESS / SLEEP DETECTED")
    print(f"  Time        : {timestamp}")
    print(f"  Eyes closed : {closed_duration:.1f} seconds")
    print(f"  PERCLOS     : {perclos * 100:.1f}%")
    print(f"  Action      : Closing video player...")
    print(f"{sep}\n")

    # ── Log file ──────────────────────────────────────────────
    logger.warning(sep)
    logger.warning("DROWSINESS / SLEEP EVENT DETECTED")
    logger.warning(f"  Timestamp        : {timestamp}")
    logger.warning(f"  Eyes closed for  : {closed_duration:.1f}s  "
                   f"(threshold: {CONSECUTIVE_CLOSED_SECONDS}s)")
    logger.warning(f"  PERCLOS score    : {perclos * 100:.1f}%  "
                   f"(threshold: {PERCLOS_DROWSY_THRESHOLD * 100:.0f}%)")
    logger.warning(sep)

    # ── Kill video player ─────────────────────────────────────
    killed = find_and_kill_video_players()
    if killed:
        msg = f"Video player(s) closed: {', '.join(killed)}"
        print(f"  [OK] {msg}\n")
        logger.warning(f"ACTION: {msg}")
    else:
        msg = "No video player process found running."
        print(f"  [INFO] {msg}\n")
        logger.info(f"ACTION: {msg}")

    logger.info("Resuming monitoring...\n" + "─" * 60)


# ─────────────────────────────────────────────
# MAIN DETECTOR
# ─────────────────────────────────────────────

def run_detector():
    if not MODEL_PATH.exists():
        logger.error(f"Model file not found: {MODEL_PATH}")
        logger.error("Download it with:")
        logger.error("  Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/"
                     "face_landmarker/face_landmarker/float16/latest/face_landmarker.task "
                     "-OutFile models/face_landmarker.task")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Drowsiness Detection System started")
    logger.info(f"  Model            : {MODEL_PATH}")
    logger.info(f"  EAR threshold    : {EAR_CLOSED_THRESHOLD}")
    logger.info(f"  PERCLOS window   : {PERCLOS_WINDOW_FRAMES} frames")
    logger.info(f"  PERCLOS trigger  : {PERCLOS_DROWSY_THRESHOLD * 100:.0f}%")
    logger.info(f"  Consecutive shut : {CONSECUTIVE_CLOSED_SECONDS}s")
    logger.info("=" * 60)

    # ── Build FaceLandmarker (Tasks API) ──────────────────────
    base_options = mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        logger.error(f"Cannot open webcam (index {WEBCAM_INDEX}). Exiting.")
        sys.exit(1)

    # State
    ear_history      = deque(maxlen=PERCLOS_WINDOW_FRAMES)
    eye_closed_start = None
    shutdown_triggered = False
    frame_idx        = 0

    print("\n" + "─" * 60)
    print("  DROWSINESS DETECTOR RUNNING")
    print("  - Keep your face visible to the webcam")
    print("  - System closes your video player when you fall asleep")
    print("  - Press 'q' in the webcam window to quit")
    print("─" * 60 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # Convert BGR → RGB for mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Detect landmarks (VIDEO mode needs monotonic timestamp)
            timestamp_ms = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            ear_l = ear_r = 0.30
            perclos        = 0.0
            closed_duration = 0.0
            state          = "ALERT"

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]   # first face

                ear_l = compute_ear(landmarks, LEFT_EYE_INDICES,  w, h)
                ear_r = compute_ear(landmarks, RIGHT_EYE_INDICES, w, h)
                avg_ear = (ear_l + ear_r) / 2.0

                eyes_closed = avg_ear < EAR_CLOSED_THRESHOLD
                ear_history.append(1 if eyes_closed else 0)

                if ear_history:
                    perclos = sum(ear_history) / len(ear_history)

                now = time.time()
                if eyes_closed:
                    if eye_closed_start is None:
                        eye_closed_start = now
                    closed_duration = now - eye_closed_start
                else:
                    eye_closed_start = None
                    closed_duration  = 0.0

                # ── Classify state ────────────────────────────────────
                if closed_duration >= CONSECUTIVE_CLOSED_SECONDS:
                    state = "SLEEPING"
                elif perclos >= PERCLOS_DROWSY_THRESHOLD:
                    state = "DROWSY"
                else:
                    state = "ALERT"

                # ── Trigger shutdown ──────────────────────────────────
                if state == "SLEEPING" and not shutdown_triggered:
                    shutdown_triggered = True
                    _trigger_shutdown(closed_duration, perclos)

            else:
                eye_closed_start = None
                ear_history.clear()
                state = "NO FACE"

            # Re-arm after a brief cooldown so demo can trigger again
            if shutdown_triggered and state == "ALERT":
                shutdown_triggered = False

            frame = draw_overlay(frame, ear_l, ear_r, perclos, closed_duration, state)
            cv2.imshow("Drowsiness Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User pressed 'q' — quitting.")
                break

            frame_idx += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        logger.info("Drowsiness Detection System stopped.")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_detector()
