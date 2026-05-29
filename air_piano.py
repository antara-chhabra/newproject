#!/usr/bin/env python3
"""
Air Piano — Play piano in the air using your webcam.

Phases:
  1. SETUP       — user moves away until full body + both hands visible
  2. CALIBRATE   — capture baseline hand height
  3. TRAINING    — learn how user presses each key (guided)
  4. PLAYING     — play with learned parameters

Controls:
  SPACE — advance phases (or confirm ready)
  R     — re-run from setup
  Q     — quit
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import os
import time
import threading
import urllib.request
from collections import deque
from typing import Optional

# ── Hand-landmarker model (Tasks API) ─────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def _ensure_model() -> str:
    if not os.path.exists(_MODEL_PATH):
        print("[Air Piano] Downloading hand-landmarker model (~2 MB)…")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[Air Piano] Model ready.")
    return _MODEL_PATH

try:
    import sounddevice as sd
    AUDIO = True
except ImportError:
    AUDIO = False
    print("[!] sounddevice not found — audio disabled.  pip install sounddevice")

SAMPLE_RATE       = 44100
CALIBRATION_SECS  = 4
TRAINING_SECS_PER_KEY = 3
PRESS_COOLDOWN    = 0.32

# ── Note frequencies ──────────────────────────────────────────────────────────
NOTES: dict[str, float] = {
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13,
    "E4": 329.63, "F4": 349.23, "F#4": 369.99, "G4": 392.00,
    "G#4": 415.30, "A4": 440.00, "A#4": 466.16, "B4": 493.88,
    "C5": 523.25,
}

WHITE = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
BLACK = ["C#4", "D#4", "F#4", "G#4", "A#4"]
ALL_KEYS = WHITE + BLACK

BLACK_POS: dict[str, float] = {
    "C#4": 0.65, "D#4": 1.65, "F#4": 3.65, "G#4": 4.65, "A#4": 5.65,
}

# MediaPipe fingertip + hand label
TIPS      = [4, 8, 12, 16, 20]
TIP_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
TIP_COLS  = [
    (0,   165, 255),    # orange  — Thumb
    (80,   80, 255),    # red     — Index
    (80,  255,  80),    # green   — Middle
    (255,  80,  80),    # blue    — Ring
    (200,   0, 200),    # purple  — Pinky
]

def _zone_col(i: int, n: int) -> tuple[int, int, int]:
    hue = int(i / n * 145)
    hsv = np.uint8([[[hue, 180, 200]]])
    b, g, r = int(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0][0]), \
              int(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0][1]), \
              int(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0][2])
    return (b, g, r)

ZONE_COLS = [_zone_col(i, len(WHITE)) for i in range(len(WHITE))]


# ── Audio ─────────────────────────────────────────────────────────────────────

def _synthesize(freq: float, dur: float = 0.75) -> np.ndarray:
    """Piano-ish ADSR tone with harmonics."""
    n = int(SAMPLE_RATE * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    wave = (
        np.sin(2 * np.pi * freq * t)       * 0.50 +
        np.sin(2 * np.pi * freq * 2 * t)   * 0.25 +
        np.sin(2 * np.pi * freq * 3 * t)   * 0.12 +
        np.sin(2 * np.pi * freq * 0.5 * t) * 0.08
    )
    a = max(1, int(0.005 * SAMPLE_RATE))
    d = max(1, int(0.040 * SAMPLE_RATE))
    r = max(1, int(0.200 * SAMPLE_RATE))
    env = np.ones(n) * 0.70
    env[:a]     = np.linspace(0,   1.00, a)
    env[a:a+d]  = np.linspace(1.0, 0.70, d)
    env[-r:]    = np.linspace(0.7, 0.0,  r)
    return (wave * env * 0.45).astype(np.float32)


class AudioEngine:
    def __init__(self):
        self._cache: dict[str, np.ndarray] = {}

    def preload(self):
        for note in NOTES:
            self._get(note)

    def _get(self, note: str) -> np.ndarray:
        if note not in self._cache:
            self._cache[note] = _synthesize(NOTES[note])
        return self._cache[note]

    def play(self, note: str):
        if not AUDIO or note not in NOTES:
            return
        data = self._get(note)
        threading.Thread(target=sd.play, args=(data, SAMPLE_RATE), daemon=True).start()


# ── Ripple effect ─────────────────────────────────────────────────────────────

class Ripple:
    def __init__(self, x: int, y: int, color: tuple[int, int, int] = (0, 210, 255)):
        self.x, self.y, self.color = x, y, color
        self._t0 = time.time()

    @property
    def alive(self) -> bool:
        return (time.time() - self._t0) < 0.65

    def draw(self, frame: np.ndarray):
        dt = time.time() - self._t0
        radius = int(8 + dt * 110)
        alpha  = max(0.0, 1.0 - dt / 0.65)
        ov = frame.copy()
        cv2.circle(ov, (self.x, self.y), radius, self.color, 3)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)


# ── Piano layout ──────────────────────────────────────────────────────────────

class PianoLayout:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.ox, self.oy, self.tw, self.th = x, y, w, h
        self.wkw  = w / len(WHITE)
        self.bkw  = self.wkw * 0.56
        self.bkh  = h * 0.62

    def white_rect(self, i: int) -> tuple[int, int, int, int]:
        x1 = self.ox + int(i * self.wkw)
        x2 = self.ox + int((i + 1) * self.wkw)
        return x1, self.oy, x2, self.oy + self.th

    def black_rect(self, note: str) -> tuple[int, int, int, int]:
        pos = BLACK_POS[note]
        x1  = self.ox + int(pos * self.wkw + (self.wkw - self.bkw) / 2)
        x2  = x1 + int(self.bkw)
        return x1, self.oy, x2, int(self.oy + self.bkh)

    def key_for_x(self, px: float) -> Optional[str]:
        rel = (px - self.ox) / self.tw
        if not (0.0 <= rel <= 1.0):
            return None
        n    = len(WHITE)
        wn   = 1.0 / n
        bwn  = wn * 0.56
        for note, pos in BLACK_POS.items():
            b0 = pos * wn + (wn - bwn) / 2
            b1 = b0 + bwn
            if b0 <= rel <= b1:
                return note
        return WHITE[min(int(rel * n), n - 1)]

    def draw(self, frame: np.ndarray, active: set[str], ripples: list[Ripple]):
        # White keys
        for i, note in enumerate(WHITE):
            x1, y1, x2, y2 = self.white_rect(i)
            col = (110, 235, 140) if note in active else (238, 238, 238)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)
            lbl = note[:-1]
            lx  = x1 + int(self.wkw / 2) - 7
            cv2.putText(frame, lbl, (lx, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)

        # Black keys
        for note in BLACK:
            x1, y1, x2, y2 = self.black_rect(note)
            col = (60, 210, 90) if note in active else (28, 28, 28)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (110, 110, 110), 1)

        for r in ripples:
            r.draw(frame)


# ── Main app ──────────────────────────────────────────────────────────────────

class AirPiano:
    SETUP      = 0
    CALIBRATE  = 1
    TRAINING   = 2
    PLAYING    = 3

    def __init__(self):
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=_ensure_model()),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.72,
            min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.60,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        self.audio = AudioEngine()

        self.phase      = self.SETUP
        self.phase_start = time.time()

        # Setup phase
        self.setup_ready = False

        # Calibration phase
        self.cal_ys: list[float] = []
        self.baseline_y = 0.0

        # Training phase
        self.training_key_idx = 0
        self.train_stats: dict[tuple[str, str], list[float]] = {}  # (hand, key) -> [depths]

        # Playing phase
        self.active:   set[str]                      = set()
        self.cooldown: dict[str, float]              = {}
        self.finger_down: dict[str, bool]            = {}
        self.ripples:  list[Ripple]                  = []
        self.history:  deque[tuple[float, str]]      = deque(maxlen=30)
        self.press_thresholds: dict[tuple[str, str], float] = {}  # (hand, key) -> threshold

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("[Air Piano] Pre-caching audio…")
        self.audio.preload()
        print("[Air Piano] Ready. Starting setup phase.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            DASH_W = 225
            CAM_W  = w - DASH_W

            # Piano at lower y (0.65 instead of 0.79)
            piano_y = int(h * 0.65)
            piano_h = h - piano_y
            layout  = PianoLayout(0, piano_y, CAM_W, piano_h)

            # Hand detection
            rgb    = cv2.cvtColor(frame[:, :CAM_W], cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res    = self.landmarker.detect_for_video(mp_img, int(time.time() * 1000))

            # Collect fingertips with hand label (left/right)
            tips: list[tuple[int, int, str, int, str]] = []  # px, py, fname, cidx, hand_label
            if res.hand_landmarks and res.handedness:
                for hidx, hand_lm in enumerate(res.hand_landmarks):
                    hand_label = res.handedness[hidx].category_name  # "Left" or "Right"
                    for i, tid in enumerate(TIPS):
                        lm = hand_lm[tid]
                        px = int(lm.x * CAM_W)
                        py = int(lm.y * h)
                        tips.append((px, py, TIP_NAMES[i], i, hand_label))

            if self.phase == self.SETUP:
                self._handle_setup(frame, tips, h, CAM_W)
            elif self.phase == self.CALIBRATE:
                self._handle_calibrate(frame, tips, h, CAM_W)
            elif self.phase == self.TRAINING:
                self._handle_training(frame, tips, h, CAM_W, layout)
            elif self.phase == self.PLAYING:
                self._handle_playing(frame, tips, h, CAM_W, layout)

            cv2.imshow("Air Piano", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord(" "):  # spacebar
                self._advance_phase()
            elif k == ord("r"):
                self.phase = self.SETUP
                self.setup_ready = False
                self.cal_ys.clear()
                self.active.clear()
                self.training_key_idx = 0
                self.train_stats.clear()
                print("[Air Piano] Reset to setup phase.")

        self.landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

    def _advance_phase(self):
        if self.phase == self.SETUP:
            if self.setup_ready:
                self.phase = self.CALIBRATE
                self.phase_start = time.time()
                self.cal_ys.clear()
                print("[Air Piano] Starting calibration…")
        elif self.phase == self.CALIBRATE:
            if self.cal_ys:
                self.baseline_y = float(np.percentile(self.cal_ys, 70))
            else:
                self.baseline_y = 999  # fallback
            print(f"[Air Piano] Baseline y = {self.baseline_y:.1f}. Starting training…")
            self.phase = self.TRAINING
            self.training_key_idx = 0
            self.train_stats.clear()
            self.phase_start = time.time()
        elif self.phase == self.TRAINING:
            self.training_key_idx += 1
            if self.training_key_idx >= len(ALL_KEYS):
                self._compute_thresholds()
                self.phase = self.PLAYING
                self.phase_start = time.time()
                print("[Air Piano] Training complete. Ready to play!")
            else:
                self.phase_start = time.time()

    def _handle_setup(self, frame: np.ndarray, tips, h: int, cam_w: int):
        """Ask user to move away until full body and both hands are visible."""
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (cam_w, h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.50, frame, 0.50, 0, frame)

        lines = [
            ("AIR  PIANO",                                    1.05, (0, 225, 130), 2),
            ("",                                               0.0,  (0, 0, 0),    0),
            ("SETUP",                                          0.75, (200, 200, 200), 1),
            ("",                                               0.0,  (0, 0, 0),    0),
            ("Please move away from the screen until your",    0.50, (175, 175, 175), 1),
            ("entire face AND both your hands are clearly",    0.50, (175, 175, 175), 1),
            ("visible in the frame.",                          0.50, (175, 175, 175), 1),
            ("",                                               0.0,  (0, 0, 0),    0),
        ]

        y0 = h // 8
        for i, (txt, sz, col, th) in enumerate(lines):
            if not txt or sz == 0.0:
                continue
            tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sz, th)[0][0]
            tx = (cam_w - tw) // 2
            cv2.putText(frame, txt, (tx, y0 + i * 46),
                        cv2.FONT_HERSHEY_SIMPLEX, sz, col, th)

        # Count hands visible
        hand_count = len(set((x, y) for x, y, *_ in tips)) // 5  # rough: 5 tips per hand
        if len(tips) >= 8:  # at least 8 fingertips = likely both hands
            self.setup_ready = True
            status = "✓ READY"
            col = (0, 255, 120)
        else:
            status = f"Hands visible: {len(tips) // 5} / 2"
            col = (0, 165, 255)

        cv2.putText(frame, status, (cam_w // 2 - 80, h // 2 + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, col, 2)
        cv2.putText(frame, "SPACE: continue    Q: quit",
                    (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 110), 1)

        # Draw fingertips
        for px, py, _, cidx, _ in tips:
            cv2.circle(frame, (px, py), 10, TIP_COLS[cidx], -1)
            cv2.circle(frame, (px, py), 12, (255, 255, 255), 2)

    def _handle_calibrate(self, frame: np.ndarray, tips, h: int, cam_w: int):
        """Calibration: capture baseline hand height."""
        elapsed = time.time() - self.phase_start
        if elapsed > CALIBRATION_SECS:
            self._advance_phase()
            return

        prog = min(1.0, elapsed / CALIBRATION_SECS)

        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (cam_w, h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)

        lines = [
            ("CALIBRATION",                      0.75, (200, 200, 200), 1),
            ("",                                 0.0,  (0, 0, 0),    0),
            ("Hold your hands in your natural",  0.50, (175, 175, 175), 1),
            ("playing position. Stay still.",    0.50, (175, 175, 175), 1),
            ("",                                 0.0,  (0, 0, 0),    0),
            (f"Recording…  {max(0.0, CALIBRATION_SECS - elapsed):.1f}s",
                                                 0.55, (0, 205, 255), 1),
        ]
        y0 = h // 7
        for i, (txt, sz, col, th) in enumerate(lines):
            if not txt or sz == 0.0:
                continue
            tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sz, th)[0][0]
            tx = (cam_w - tw) // 2
            cv2.putText(frame, txt, (tx, y0 + i * 46),
                        cv2.FONT_HERSHEY_SIMPLEX, sz, col, th)

        # Progress bar
        bw = int(cam_w * 0.55)
        bx = (cam_w - bw) // 2
        by = h // 2 + 40
        cv2.rectangle(frame, (bx, by), (bx + bw, by + 18), (45, 45, 55), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(bw * prog), by + 18), (0, 215, 120), -1)

        # Collect y positions
        for px, py, _, cidx, _ in tips:
            self.cal_ys.append(float(py))
            cv2.circle(frame, (px, py), 10, TIP_COLS[cidx], -1)
            cv2.circle(frame, (px, py), 12, (255, 255, 255), 2)

        cv2.putText(frame, "R: reset    Q: quit",
                    (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 110), 1)

    def _handle_training(self, frame: np.ndarray, tips, h: int, cam_w: int, layout: PianoLayout):
        """Training: learn how user presses each key."""
        elapsed = time.time() - self.phase_start
        if elapsed > TRAINING_SECS_PER_KEY:
            self._advance_phase()
            return

        current_key = ALL_KEYS[self.training_key_idx]
        prog = min(1.0, elapsed / TRAINING_SECS_PER_KEY)

        # Draw zones
        ov = frame.copy()
        zone_top = max(0, int(self.baseline_y) - 5)
        zone_bot = layout.oy
        if zone_bot > zone_top:
            for i, note in enumerate(WHITE):
                x1, _, x2, _ = layout.white_rect(i)
                cv2.rectangle(ov, (x1, zone_top), (x2, zone_bot), ZONE_COLS[i], -1)
            cv2.addWeighted(ov, 0.16, frame, 0.84, 0, frame)

        # Baseline
        bl = int(self.baseline_y)
        cv2.line(frame, (0, bl), (cam_w, bl), (0, 235, 200), 2)

        # Draw piano
        layout.draw(frame, {current_key}, [])

        # Instruction overlay
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (cam_w, h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.35, frame, 0.65, 0, frame)

        lines = [
            ("TRAINING",                        0.70, (200, 200, 200), 1),
            ("",                                 0.0,  (0, 0, 0),    0),
            (f"Press key: {current_key}",       1.00, (0, 220, 200), 2),
            ("",                                 0.0,  (0, 0, 0),    0),
            (f"({self.training_key_idx + 1} / {len(ALL_KEYS)})",
                                                0.50, (150, 150, 180), 1),
        ]
        y0 = h // 5
        for i, (txt, sz, col, th) in enumerate(lines):
            if not txt or sz == 0.0:
                continue
            tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sz, th)[0][0]
            tx = (cam_w - tw) // 2
            cv2.putText(frame, txt, (tx, y0 + i * 50),
                        cv2.FONT_HERSHEY_SIMPLEX, sz, col, th)

        # Progress
        bw = int(cam_w * 0.55)
        bx = (cam_w - bw) // 2
        by = h // 2 + 60
        cv2.rectangle(frame, (bx, by), (bx + bw, by + 18), (45, 45, 55), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(bw * prog), by + 18), (0, 215, 120), -1)

        # Process presses
        now = time.time()
        for px, py, fname, cidx, hand in tips:
            key = layout.key_for_x(px)
            if key == current_key:
                depth = py - bl
                if depth >= 0:
                    stat_key = (hand, current_key)
                    if stat_key not in self.train_stats:
                        self.train_stats[stat_key] = []
                    self.train_stats[stat_key].append(float(depth))

            # Draw fingertip
            cv2.circle(frame, (px, py), 11, TIP_COLS[cidx], -1)
            cv2.circle(frame, (px, py), 13, (255, 255, 255), 2)

            # Depth bar
            depth = py - bl
            if 0 < depth < 100 and key == current_key:
                intensity = min(1.0, depth / 50)
                bar_col   = tuple(int(c * intensity) for c in TIP_COLS[cidx])
                cv2.line(frame, (px, bl), (px, py), bar_col, 4)

        cv2.putText(frame, "SPACE: next key    Q: quit",
                    (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 110), 1)

    def _handle_playing(self, frame: np.ndarray, tips, h: int, cam_w: int, layout: PianoLayout):
        """Normal play mode using learned parameters."""
        now = time.time()
        new_active: set[str] = set()

        # Draw zones
        ov = frame.copy()
        zone_top = max(0, int(self.baseline_y) - 5)
        zone_bot = layout.oy
        if zone_bot > zone_top:
            for i, note in enumerate(WHITE):
                x1, _, x2, _ = layout.white_rect(i)
                cv2.rectangle(ov, (x1, zone_top), (x2, zone_bot), ZONE_COLS[i], -1)
            cv2.addWeighted(ov, 0.16, frame, 0.84, 0, frame)

        # Baseline
        bl = int(self.baseline_y)
        cv2.line(frame, (0, bl), (cam_w, bl), (0, 235, 200), 2)

        # Process fingertips
        for px, py, fname, cidx, hand in tips:
            col  = TIP_COLS[cidx]
            key  = layout.key_for_x(px)
            depth = py - bl

            # Get threshold for this (hand, key) pair
            thresh = self.press_thresholds.get((hand, key), 20)

            pressed_now = depth >= thresh
            if pressed_now and key:
                new_active.add(key)
                cd = self.cooldown.get(key, 0.0)
                if now - cd > PRESS_COOLDOWN:
                    self.cooldown[key] = now
                    self.audio.play(key)
                    self.history.appendleft((now, key))

            # Draw fingertip
            cv2.circle(frame, (px, py), 11, col, -1)
            cv2.circle(frame, (px, py), 13, (255, 255, 255), 2)

            # Depth bar
            if 0 < depth < thresh * 3 and key:
                intensity = min(1.0, depth / thresh)
                bar_col   = tuple(int(c * intensity) for c in col)
                cv2.line(frame, (px, bl), (px, py), bar_col, 4)
                glow_r = int(11 + intensity * 14)
                ov = frame.copy()
                cv2.circle(ov, (px, py), glow_r, col, 2)
                cv2.addWeighted(ov, intensity * 0.7, frame, 1 - intensity * 0.7, 0, frame)

        self.active = new_active
        layout.draw(frame, self.active, [])
        self._draw_dashboard(frame, h, cam_w, layout)

        cv2.putText(frame, "R: recalibrate    Q: quit",
                    (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 110), 1)

    def _compute_thresholds(self):
        """Compute per-hand per-key press thresholds from training data."""
        for (hand, key), depths in self.train_stats.items():
            if depths:
                # Use 40th percentile to capture "normal" press depth
                threshold = float(np.percentile(depths, 40))
                # Clamp between 8 and 60 px
                threshold = max(8.0, min(60.0, threshold))
                self.press_thresholds[(hand, key)] = threshold
                print(f"  {hand} {key}: threshold = {threshold:.1f} px")

    def _draw_dashboard(self, frame: np.ndarray, h: int, cam_w: int, layout: PianoLayout):
        w = frame.shape[1]
        DASH_W = w - cam_w

        # Background
        ov = frame.copy()
        cv2.rectangle(ov, (cam_w, 0), (w, h), (14, 17, 24), -1)
        cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)

        dx = cam_w + 10
        dy = 26

        # Title
        cv2.putText(frame, "AIR PIANO", (dx, dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 130), 2)
        dy += 8
        cv2.line(frame, (cam_w + 2, dy), (w - 2, dy), (48, 52, 68), 1)
        dy += 18

        # NOW PLAYING
        cv2.putText(frame, "NOW PLAYING", (dx, dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 155), 1)
        dy += 19
        act_str = "  ".join(sorted(self.active)) if self.active else u"—"
        cv2.putText(frame, act_str[:22], (dx, dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 120), 1)
        dy += 24
        cv2.line(frame, (cam_w + 2, dy), (w - 2, dy), (48, 52, 68), 1)
        dy += 12

        # Mini keyboard
        cv2.putText(frame, "KEYBOARD", (dx, dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 155), 1)
        dy += 7
        mk_w = DASH_W - 14
        mk_h = 38
        mk_x = cam_w + 7
        n    = len(WHITE)
        wkw  = mk_w / n
        bkw  = wkw * 0.56
        bkh  = mk_h * 0.62

        for i, note in enumerate(WHITE):
            kx = int(mk_x + i * wkw)
            col = (110, 235, 140) if note in self.active else (215, 215, 215)
            cv2.rectangle(frame, (kx, dy), (kx + int(wkw) - 1, dy + mk_h), col, -1)
            cv2.rectangle(frame, (kx, dy), (kx + int(wkw) - 1, dy + mk_h), (70, 70, 70), 1)

        for note, pos in BLACK_POS.items():
            kx  = int(mk_x + pos * wkw + (wkw - bkw) / 2)
            col = (60, 200, 90) if note in self.active else (28, 28, 28)
            cv2.rectangle(frame, (kx, dy), (kx + int(bkw), dy + int(bkh)), col, -1)

        dy += mk_h + 16
        cv2.line(frame, (cam_w + 2, dy - 2), (w - 2, dy - 2), (48, 52, 68), 1)

        # Note history
        cv2.putText(frame, "RECENT NOTES", (dx, dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 155), 1)
        dy += 18
        now = time.time()
        for ts, note in list(self.history)[:14]:
            age  = now - ts
            if age > 9:
                continue
            fade = max(0.12, 1.0 - age / 7.0)
            v    = int(255 * fade)
            bg_w = int((DASH_W - 70) * fade)
            cv2.rectangle(frame, (dx, dy - 10), (dx + bg_w, dy - 2),
                          (0, int(v * 0.35), 0), -1)
            cv2.putText(frame, f"{note:<6} {age:.1f}s", (dx, dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, v, int(v * 0.5)), 1)
            dy += 17
            if dy > h - 45:
                break

        # Baseline
        cv2.line(frame, (cam_w + 2, h - 38), (w - 2, h - 38), (48, 52, 68), 1)
        cv2.putText(frame, f"baseline y={int(self.baseline_y)}",
                    (dx, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 100), 1)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Air Piano – Two-Hand Training Edition")
    print("=" * 60)
    print(f"  Audio : {'sounddevice (ON)' if AUDIO else 'DISABLED (pip install sounddevice)'}")
    print("  Keys  : C4 D4 E4 F4 G4 A4 B4 C5 (+ sharps/flats)")
    print()
    print("  PHASES")
    print("  1. SETUP       — move away until face + both hands visible")
    print("  2. CALIBRATE   — establish baseline resting hand height")
    print("  3. TRAINING    — press each key in sequence (we learn your style)")
    print("  4. PLAYING     — play freely with personalized thresholds")
    print()
    print("  CONTROLS")
    print("  SPACE — advance to next phase / next training key")
    print("  R     — reset to setup")
    print("  Q     — quit")
    print()
    AirPiano().run()


if __name__ == "__main__":
    main()
