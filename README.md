# Drowsiness Detection System

Real-time drowsiness monitoring via webcam. When sleep is detected, it **closes the video player**, prints a message to the terminal, and logs the event.

## How it Works

```
Webcam → MediaPipe Face Mesh → EAR (Eye Aspect Ratio) → PERCLOS score
                                                                ↓
                                                   Eyes closed > 2.5s?
                                                                ↓
                                                    ┌─── YES ───┐
                                                    │           │
                                             Print to     Write to
                                             terminal     logs/drowsiness.log
                                                    │           │
                                                    └─── Kill video player ──┘
```

### Detection Logic (Two Conditions, Either Triggers)

| Rule | Condition | Meaning |
|------|-----------|---------|
| **Consecutive closure** | Eyes closed ≥ **2.5 seconds** | Hard sleep indicator |
| **PERCLOS** | > **40%** of last 60 frames eyes closed | Accumulated drowsiness |

EAR (Eye Aspect Ratio) < **0.20** → eye counted as closed.

---

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `opencv-python`, `mediapipe`, `numpy`, `scipy`

---

## Run

### Option A — Detector only (open your video player manually first)
```bash
python detector.py
```

### Option B — Auto-launch VLC + detector
```bash
python demo.py                        # launches VLC, then detector
python demo.py path/to/movie.mp4      # opens the video in VLC first
```

---

## What Happens When Sleep is Detected

**Terminal output:**
```
============================================================
  🚨 DROWSINESS / SLEEP DETECTED
  Time        : 2026-09-28 21:30:15
  Eyes closed : 2.6 seconds
  PERCLOS     : 48.3%
  Action      : Closing video player...
============================================================

  ✅ Video player(s) closed: vlc.exe
```

**Log file (`logs/drowsiness.log`):**
```
2026-09-28 21:30:15,412 | WARNING | ============================
2026-09-28 21:30:15,412 | WARNING | DROWSINESS / SLEEP EVENT DETECTED
2026-09-28 21:30:15,413 | WARNING |   Timestamp        : 2026-09-28 21:30:15
2026-09-28 21:30:15,413 | WARNING |   Eyes closed for  : 2.6s  (threshold: 2.5s)
2026-09-28 21:30:15,413 | WARNING |   PERCLOS score    : 48.3%  (threshold: 40%)
2026-09-28 21:30:15,414 | WARNING | ACTION: Video player(s) closed: vlc.exe
```

---

## Video Players Detected & Closed

The system auto-detects and kills any of these processes:
- VLC (`vlc.exe`)
- MPV (`mpv.exe`)
- MPC-HC (`mpc-hc64.exe`, `mpc-hc.exe`)
- Windows Media Player (`wmplayer.exe`)
- Windows Movies & TV (`Movies.UI.exe`)
- PotPlayer (`PotPlayerMini64.exe`, `PotPlayerMini.exe`)

---

## Config (top of `detector.py`)

```python
EAR_CLOSED_THRESHOLD        = 0.20   # EAR below this = eye closed
PERCLOS_WINDOW_FRAMES       = 60     # evaluate over last 60 frames (~2s at 30fps)
PERCLOS_DROWSY_THRESHOLD    = 0.40   # 40% closed frames = drowsy
CONSECUTIVE_CLOSED_SECONDS  = 2.5   # 2.5s continuous closure = sleeping
```

---

## Project Structure

```
vlc-drowsiness/
├── detector.py        # Main detection logic
├── demo.py            # Demo launcher (opens VLC + detector)
├── requirements.txt   # Dependencies
├── README.md
└── logs/
    └── drowsiness.log # Auto-created on first detection
```
