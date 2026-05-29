#!/usr/bin/env python3
"""
Air Piano — Play piano in the air using your webcam.

Phases (fully automatic — no keyboard required):
  1. SETUP      — step back until face + both hands are clearly visible
  2. CALIBRATE  — hover hands at playing height so we can measure your baseline
  3. TRAINING   — press randomly ordered keys so we learn your press depth
  4. PLAYING    — play freely with your personalized settings

Controls:
  R  — restart from the beginning
  Q  — quit
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import os
import random
import time
import threading
import urllib.request
from collections import deque
from typing import Optional

# ── MediaPipe model ───────────────────────────────────────────────────────────
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


# ── Audio ─────────────────────────────────────────────────────────────────────
try:
    import sounddevice as sd
    AUDIO = True
except ImportError:
    AUDIO = False
    print("[!] sounddevice not found — audio disabled.  pip install sounddevice")

SAMPLE_RATE = 44100

# ── Notes ─────────────────────────────────────────────────────────────────────
NOTES: dict[str, float] = {
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13,
    "E4": 329.63, "F4": 349.23, "F#4": 369.99, "G4": 392.00,
    "G#4": 415.30, "A4": 440.00, "A#4": 466.16, "B4": 493.88,
    "C5": 523.25,
}

WHITE    = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
BLACK    = ["C#4", "D#4", "F#4", "G#4", "A#4"]
ALL_KEYS = WHITE + BLACK   # 13 total

BLACK_POS: dict[str, float] = {
    "C#4": 0.65, "D#4": 1.65, "F#4": 3.65, "G#4": 4.65, "A#4": 5.65,
}

# ── Fingertip constants ───────────────────────────────────────────────────────
TIPS      = [4, 8, 12, 16, 20]
TIP_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
TIP_COLS  = [
    (0,   165, 255),   # orange  — Thumb
    (80,   80, 255),   # red     — Index
    (80,  255,  80),   # green   — Middle
    (255,  80,  80),   # blue    — Ring
    (200,   0, 200),   # purple  — Pinky
]

def _zone_col(i: int, n: int) -> tuple[int, int, int]:
    hue = int(i / n * 145)
    hsv = np.uint8([[[hue, 180, 200]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

ZONE_COLS = [_zone_col(i, len(WHITE)) for i in range(len(WHITE))]

# ── Phase timing ──────────────────────────────────────────────────────────────
SETUP_HANDS_SECS      = 1.5   # seconds both hands must be visible to auto-advance
CALIBRATION_SECS      = 4     # seconds to collect fingertip y-positions
TRAINING_SECS_PER_KEY = 4     # max seconds per training key
TRAINING_MIN_PRESSES  = 3     # auto-advance early once user has pressed this many times
PRESS_COOLDOWN        = 0.32  # minimum seconds between same-key triggers in play mode


# ── Audio synthesis ───────────────────────────────────────────────────────────

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
    env[:a]    = np.linspace(0,   1.00, a)
    env[a:a+d] = np.linspace(1.0, 0.70, d)
    env[-r:]   = np.linspace(0.7, 0.0,  r)
    return (wave * env * 0.45).astype(np.float32)


class AudioEngine:
    """
    Callback-based real-time audio mixer.

    Each note is pre-synthesized into a float32 numpy array and cached.
    When play() is called, the array is added to a list of active voices.
    The OutputStream callback runs on a dedicated audio thread, mixing all
    active voices into a single output buffer each block (512 samples).
    This avoids the crackling and dropped notes you get from calling
    sd.play() from multiple threads, where each call interrupts the last.
    """

    def __init__(self):
        self._cache:  dict[str, np.ndarray] = {}
        self._voices: list[list]             = []   # each: [ndarray, int cursor]
        self._lock   = threading.Lock()
        self._stream: Optional["sd.OutputStream"] = None

        if AUDIO:
            self._stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._callback,
                blocksize=512,
            )
            self._stream.start()

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status):
        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            remaining = []
            for voice in self._voices:
                data, pos = voice
                end   = min(pos + frames, len(data))
                chunk = data[pos:end]
                out[:len(chunk)] += chunk
                if end < len(data):
                    voice[1] = end
                    remaining.append(voice)
            self._voices = remaining
        np.clip(out, -1.0, 1.0, out=out)
        outdata[:, 0] = out

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
        with self._lock:
            self._voices.append([self._get(note), 0])

    def close(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()


# ── Visual effects ────────────────────────────────────────────────────────────

class Ripple:
    def __init__(self, x: int, y: int, color: tuple[int, int, int] = (0, 210, 255)):
        self.x, self.y, self.color = x, y, color
        self._t0 = time.time()

    @property
    def alive(self) -> bool:
        return (time.time() - self._t0) < 0.65

    def draw(self, frame: np.ndarray):
        dt     = time.time() - self._t0
        radius = int(8 + dt * 110)
        alpha  = max(0.0, 1.0 - dt / 0.65)
        ov = frame.copy()
        cv2.circle(ov, (self.x, self.y), radius, self.color, 3)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)


# ── Piano layout ──────────────────────────────────────────────────────────────

class PianoLayout:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.ox, self.oy, self.tw, self.th = x, y, w, h
        self.wkw = w / len(WHITE)
        self.bkw = self.wkw * 0.56
        self.bkh = h * 0.62

    def white_rect(self, i: int) -> tuple[int, int, int, int]:
        x1 = self.ox + int(i * self.wkw)
        x2 = self.ox + int((i + 1) * self.wkw)
        return x1, self.oy, x2, self.oy + self.th

    def black_rect(self, note: str) -> tuple[int, int, int, int]:
        pos = BLACK_POS[note]
        x1  = self.ox + int(pos * self.wkw + (self.wkw - self.bkw) / 2)
        return x1, self.oy, x1 + int(self.bkw), int(self.oy + self.bkh)

    def key_for_x(self, px: float) -> Optional[str]:
        rel = (px - self.ox) / self.tw
        if not (0.0 <= rel <= 1.0):
            return None
        n   = len(WHITE)
        wn  = 1.0 / n
        bwn = wn * 0.56
        for note, pos in BLACK_POS.items():
            b0 = pos * wn + (wn - bwn) / 2
            if b0 <= rel <= b0 + bwn:
                return note
        return WHITE[min(int(rel * n), n - 1)]

    def draw(self, frame: np.ndarray, active: set[str],
             ripples: list[Ripple], highlight: Optional[str] = None):
        # White keys
        for i, note in enumerate(WHITE):
            x1, y1, x2, y2 = self.white_rect(i)
            if note in active:        col = (110, 235, 140)
            elif note == highlight:   col = (0,   220, 255)
            else:                     col = (238, 238, 238)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)
            lx = x1 + int(self.wkw / 2) - 7
            cv2.putText(frame, note[:-1], (lx, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)

        # Black keys
        for note in BLACK:
            x1, y1, x2, y2 = self.black_rect(note)
            if note in active:        col = (60,  210,  90)
            elif note == highlight:   col = (0,   180, 200)
            else:                     col = (28,   28,  28)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (110, 110, 110), 1)

        for r in ripples:
            r.draw(frame)


# ── Main application ──────────────────────────────────────────────────────────

class AirPiano:
    SETUP     = 0
    CALIBRATE = 1
    TRAINING  = 2
    PLAYING   = 3

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
        self.audio      = AudioEngine()

        self.phase       = self.SETUP
        self.phase_start = time.time()

        # ── SETUP ────────────────────────────────────────────────────────────
        self._setup_both_since: Optional[float] = None

        # ── CALIBRATE ────────────────────────────────────────────────────────
        # Each frame we record the y-pixel of every detected fingertip.
        # After CALIBRATION_SECS seconds we take the 70th percentile of all
        # those samples.  That pixel row becomes baseline_y — the screen height
        # at which the user's fingertips naturally rest while hovering.
        # Pressing a finger below (baseline_y + threshold) triggers a note.
        self.cal_ys:    list[float] = []
        self.baseline_y = 0.0

        # ── TRAINING ─────────────────────────────────────────────────────────
        # Keys are shown in a random order.  Each time the user presses the
        # target key we record how many pixels below baseline their fingertip
        # reached.  After training we use the 40th-percentile depth per
        # (hand, key) pair as that combination's personal press threshold.
        self.training_order:   list[str]                         = []
        self.training_key_idx: int                               = 0
        self.training_presses: int                               = 0
        self.train_depths:     dict[tuple[str,str], list[float]] = {}
        self.finger_down:      dict[str, bool]                   = {}

        # ── PLAYING ───────────────────────────────────────────────────────────
        self.active:           set[str]                      = set()
        self.cooldown:         dict[str, float]              = {}
        self.press_thresholds: dict[tuple[str,str], float]   = {}
        self.ripples:          list[Ripple]                  = []
        self.history:          deque[tuple[float, str]]      = deque(maxlen=30)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("[Air Piano] Pre-caching audio…")
        self.audio.preload()
        print("[Air Piano] Ready.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # Piano spans the full frame width, sits in the lower 28%
            piano_y = int(h * 0.72)
            piano_h = h - piano_y
            layout  = PianoLayout(0, piano_y, w, piano_h)

            # Hand detection over the entire frame
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res    = self.landmarker.detect_for_video(mp_img, int(time.time() * 1000))

            # Build tip list: (px, py, finger_name, color_idx, hand_label)
            tips: list[tuple[int, int, str, int, str]] = []
            if res.hand_landmarks and res.handedness:
                for hidx, hand_lm in enumerate(res.hand_landmarks):
                    hand_label = res.handedness[hidx][0].category_name  # "Left" / "Right"
                    for i, tid in enumerate(TIPS):
                        lm = hand_lm[tid]
                        tips.append((int(lm.x * w), int(lm.y * h),
                                     TIP_NAMES[i], i, hand_label))

            if   self.phase == self.SETUP:     self._phase_setup(frame, tips, h, w)
            elif self.phase == self.CALIBRATE: self._phase_calibrate(frame, tips, h, w)
            elif self.phase == self.TRAINING:  self._phase_training(frame, tips, h, w, layout)
            else:                              self._phase_playing(frame, tips, h, w, layout)

            cv2.imshow("Air Piano", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("r"):
                self._reset()

        self.audio.close()
        self.landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

    def _reset(self):
        self.phase             = self.SETUP
        self.phase_start       = time.time()
        self._setup_both_since = None
        self.cal_ys.clear()
        self.training_order.clear()
        self.training_key_idx  = 0
        self.training_presses  = 0
        self.train_depths.clear()
        self.finger_down.clear()
        self.active.clear()
        self.press_thresholds.clear()
        self.ripples.clear()
        print("[Air Piano] Reset.")

    # ── Phase 1: SETUP ────────────────────────────────────────────────────────

    def _phase_setup(self, frame: np.ndarray, tips, h: int, w: int):
        """Auto-advances once both hands are visible for SETUP_HANDS_SECS."""
        both = len(tips) >= 8  # ≥8 fingertips → likely both hands

        if both:
            if self._setup_both_since is None:
                self._setup_both_since = time.time()
            held = time.time() - self._setup_both_since
            if held >= SETUP_HANDS_SECS:
                self._start_calibrate()
                return
        else:
            self._setup_both_since = None
            held = 0.0

        # Dark overlay
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.50, frame, 0.50, 0, frame)

        for i, (txt, sz, col, th) in enumerate([
            ("AIR  PIANO",                          1.05, (0, 225, 130), 2),
            ("",                                     0.0, (0,0,0), 0),
            ("Step back until your face",            0.55, (175,175,175), 1),
            ("and both hands are clearly",           0.55, (175,175,175), 1),
            ("visible in the frame.",                0.55, (175,175,175), 1),
        ]):
            if not txt or sz == 0.0: continue
            tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sz, th)[0][0]
            cv2.putText(frame, txt, ((w - tw) // 2, h // 7 + i * 50),
                        cv2.FONT_HERSHEY_SIMPLEX, sz, col, th)

        if both:
            prog   = min(1.0, held / SETUP_HANDS_SECS)
            status = "Both hands detected — hold still…"
            scol   = (0, 255, 120)
            bw = int(w * 0.35);  bx = (w - bw) // 2;  by = h // 2 + 80
            cv2.rectangle(frame, (bx, by), (bx + bw, by + 14), (45,45,55), -1)
            cv2.rectangle(frame, (bx, by), (bx + int(bw * prog), by + 14), (0,215,120), -1)
        else:
            status = f"Hands visible: {len(tips) // 5} / 2"
            scol   = (0, 165, 255)

        tw = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0][0]
        cv2.putText(frame, status, ((w - tw) // 2, h // 2 + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, scol, 1)

        for px, py, _, cidx, _ in tips:
            cv2.circle(frame, (px, py), 10, TIP_COLS[cidx], -1)
            cv2.circle(frame, (px, py), 12, (255,255,255), 2)

        cv2.putText(frame, "R: restart    Q: quit",
                    (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90,90,110), 1)

    def _start_calibrate(self):
        self.phase       = self.CALIBRATE
        self.phase_start = time.time()
        self.cal_ys.clear()
        print("[Air Piano] Calibrating…")

    # ── Phase 2: CALIBRATE ────────────────────────────────────────────────────

    def _phase_calibrate(self, frame: np.ndarray, tips, h: int, w: int):
        """
        Sample every fingertip's y-pixel position for CALIBRATION_SECS seconds.
        Take the 70th percentile → baseline_y.
        This pixel row is the "resting height" of the user's hands.
        Pressing a finger below baseline_y + threshold triggers a note.
        Auto-advances when the timer expires.
        """
        elapsed = time.time() - self.phase_start
        if elapsed >= CALIBRATION_SECS:
            self._finish_calibrate(h)
            return

        prog = elapsed / CALIBRATION_SECS

        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)

        for i, (txt, sz, col, th) in enumerate([
            ("CALIBRATING",                           0.75, (200,200,200), 1),
            ("",                                       0.0, (0,0,0), 0),
            ("Hold your hands at the height",          0.50, (175,175,175), 1),
            ("you'd naturally play a piano.",          0.50, (175,175,175), 1),
            ("",                                       0.0, (0,0,0), 0),
            ("Measuring your hand height…",            0.50, (0,205,255), 1),
            (f"{max(0.0, CALIBRATION_SECS - elapsed):.1f}s", 0.45, (140,140,170), 1),
        ]):
            if not txt or sz == 0.0: continue
            tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sz, th)[0][0]
            cv2.putText(frame, txt, ((w - tw) // 2, h // 7 + i * 46),
                        cv2.FONT_HERSHEY_SIMPLEX, sz, col, th)

        bw = int(w * 0.50);  bx = (w - bw) // 2;  by = h // 2 + 50
        cv2.rectangle(frame, (bx, by), (bx + bw, by + 18), (45,45,55), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(bw * prog), by + 18), (0,215,120), -1)

        for px, py, _, cidx, _ in tips:
            self.cal_ys.append(float(py))
            cv2.circle(frame, (px, py), 10, TIP_COLS[cidx], -1)
            cv2.circle(frame, (px, py), 12, (255,255,255), 2)

    def _finish_calibrate(self, h: int):
        self.baseline_y = float(np.percentile(self.cal_ys, 70)) if self.cal_ys else h * 0.60
        print(f"[Air Piano] Baseline y = {self.baseline_y:.1f} px")
        self._start_training()

    # ── Phase 3: TRAINING ─────────────────────────────────────────────────────

    def _start_training(self):
        self.training_order   = random.sample(ALL_KEYS, len(ALL_KEYS))
        self.training_key_idx = 0
        self.training_presses = 0
        self.train_depths.clear()
        self.finger_down.clear()
        self.phase       = self.TRAINING
        self.phase_start = time.time()
        print(f"[Air Piano] Training order: {self.training_order}")

    def _phase_training(self, frame: np.ndarray, tips, h: int, w: int, layout: PianoLayout):
        if self.training_key_idx >= len(self.training_order):
            self._finish_training()
            return

        elapsed    = time.time() - self.phase_start
        target_key = self.training_order[self.training_key_idx]
        time_up    = elapsed >= TRAINING_SECS_PER_KEY
        enough     = self.training_presses >= TRAINING_MIN_PRESSES

        if time_up or enough:
            self.training_key_idx += 1
            self.training_presses  = 0
            self.finger_down.clear()
            self.phase_start = time.time()
            return

        bl   = int(self.baseline_y)
        prog = elapsed / TRAINING_SECS_PER_KEY

        # Zone bands
        ov = frame.copy()
        zone_top = max(0, bl - 5);  zone_bot = layout.oy
        if zone_bot > zone_top:
            for i, note in enumerate(WHITE):
                x1, _, x2, _ = layout.white_rect(i)
                cv2.rectangle(ov, (x1, zone_top), (x2, zone_bot), ZONE_COLS[i], -1)
            cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)

        cv2.line(frame, (0, bl), (w, bl), (0, 235, 200), 2)
        layout.draw(frame, set(), [], highlight=target_key)

        # Dim upper area for text
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, int(h * 0.58)), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.38, frame, 0.62, 0, frame)

        for i, (txt, sz, col, th) in enumerate([
            ("TRAINING",                                    0.70, (200,200,200), 1),
            ("",                                             0.0, (0,0,0), 0),
            (f"Press   {target_key}",                       1.10, (0,220,200), 2),
            ("",                                             0.0, (0,0,0), 0),
            (f"Key {self.training_key_idx+1} / {len(self.training_order)}"
             f"   •   {self.training_presses} press{'es' if self.training_presses!=1 else ''}",
                                                            0.48, (150,150,180), 1),
        ]):
            if not txt or sz == 0.0: continue
            tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sz, th)[0][0]
            cv2.putText(frame, txt, ((w - tw) // 2, h // 9 + i * 56),
                        cv2.FONT_HERSHEY_SIMPLEX, sz, col, th)

        # Timer bar
        bw = int(w * 0.50);  bx = (w - bw) // 2;  by = h // 2 + 30
        cv2.rectangle(frame, (bx, by), (bx + bw, by + 12), (45,45,55), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(bw * prog), by + 12), (0,215,120), -1)

        for px, py, fname, cidx, hand in tips:
            key   = layout.key_for_x(px)
            depth = py - bl
            fkey  = f"{hand}_{fname}"
            pressed_now = depth >= 10
            was_down    = self.finger_down.get(fkey, False)

            if pressed_now and key == target_key and not was_down:
                self.training_presses += 1
                self.train_depths.setdefault((hand, target_key), []).append(float(depth))
                self.audio.play(target_key)

            self.finger_down[fkey] = pressed_now
            cv2.circle(frame, (px, py), 11, TIP_COLS[cidx], -1)
            cv2.circle(frame, (px, py), 13, (255,255,255), 2)
            if 0 < depth < 80 and key == target_key:
                intensity = min(1.0, depth / 40)
                cv2.line(frame, (px, bl), (px, py),
                         tuple(int(c * intensity) for c in TIP_COLS[cidx]), 4)

    def _finish_training(self):
        self._compute_thresholds()
        self.phase       = self.PLAYING
        self.phase_start = time.time()
        self.finger_down.clear()
        self.active.clear()
        print("[Air Piano] Training complete — playing!")

    def _compute_thresholds(self):
        """
        For each (hand, key) pair, take the 40th-percentile press depth
        recorded during training as that combination's trigger threshold.
        The 40th percentile captures a light-but-intentional press rather
        than the user's deepest push.  Any (hand, key) with no training
        data falls back to 20 px.
        """
        for (hand, key), depths in self.train_depths.items():
            thresh = max(8.0, min(60.0, float(np.percentile(depths, 40))))
            self.press_thresholds[(hand, key)] = thresh
            print(f"  {hand:<6} {key:<5}  threshold = {thresh:.1f} px")

    # ── Phase 4: PLAYING ──────────────────────────────────────────────────────

    def _phase_playing(self, frame: np.ndarray, tips, h: int, w: int, layout: PianoLayout):
        now = time.time()
        bl  = int(self.baseline_y)
        new_active: set[str] = set()

        # Zone bands
        ov = frame.copy()
        zone_top = max(0, bl - 5);  zone_bot = layout.oy
        if zone_bot > zone_top:
            for i, note in enumerate(WHITE):
                x1, _, x2, _ = layout.white_rect(i)
                cv2.rectangle(ov, (x1, zone_top), (x2, zone_bot), ZONE_COLS[i], -1)
            cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)

        cv2.line(frame, (0, bl), (w, bl), (0, 235, 200), 2)

        for px, py, fname, cidx, hand in tips:
            col    = TIP_COLS[cidx]
            key    = layout.key_for_x(px)
            depth  = py - bl
            fkey   = f"{hand}_{fname}"
            thresh = self.press_thresholds.get((hand, key), 20)

            pressed_now = depth >= thresh
            was_down    = self.finger_down.get(fkey, False)

            if pressed_now and key:
                new_active.add(key)
                if not was_down:   # rising edge only — don't retrigger while held
                    cd = self.cooldown.get(key, 0.0)
                    if now - cd > PRESS_COOLDOWN:
                        self.cooldown[key] = now
                        self.audio.play(key)
                        self.history.appendleft((now, key))
                        self.ripples.append(Ripple(px, bl, col))

            self.finger_down[fkey] = pressed_now
            cv2.circle(frame, (px, py), 11, col, -1)
            cv2.circle(frame, (px, py), 13, (255,255,255), 2)

            if 0 < depth < thresh * 3 and key:
                intensity = min(1.0, depth / thresh)
                cv2.line(frame, (px, bl), (px, py),
                         tuple(int(c * intensity) for c in col), 4)
                ov2 = frame.copy()
                cv2.circle(ov2, (px, py), int(11 + intensity * 14), col, 2)
                cv2.addWeighted(ov2, intensity * 0.7, frame, 1 - intensity * 0.7, 0, frame)

        self.active  = new_active
        self.ripples = [r for r in self.ripples if r.alive]
        layout.draw(frame, self.active, self.ripples)
        self._draw_stats(frame, h, w)

        cv2.putText(frame, "R: restart    Q: quit",
                    (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80,80,100), 1)

    # ── Stats overlay (top-right corner) ─────────────────────────────────────

    def _draw_stats(self, frame: np.ndarray, h: int, w: int):
        BOX_W, BOX_H = 205, 175
        bx, by = w - BOX_W - 10, 10

        ov = frame.copy()
        cv2.rectangle(ov, (bx, by), (bx + BOX_W, by + BOX_H), (12, 15, 22), -1)
        cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (bx, by), (bx + BOX_W, by + BOX_H), (48, 52, 68), 1)

        dx, dy = bx + 8, by + 18
        cv2.putText(frame, "AIR PIANO", (dx, dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 130), 1)
        dy += 6
        cv2.line(frame, (bx+4, dy), (bx+BOX_W-4, dy), (48,52,68), 1)
        dy += 15

        cv2.putText(frame, "NOW PLAYING", (dx, dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (110,110,145), 1)
        dy += 15
        act = "  ".join(sorted(self.active)) if self.active else u"—"
        cv2.putText(frame, act[:18], (dx, dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,120), 1)
        dy += 17
        cv2.line(frame, (bx+4, dy-2), (bx+BOX_W-4, dy-2), (48,52,68), 1)

        cv2.putText(frame, "RECENT", (dx, dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (110,110,145), 1)
        dy += 15
        now = time.time()
        for ts, note in list(self.history)[:6]:
            age  = now - ts
            if age > 8: continue
            fade = max(0.2, 1.0 - age / 6.0)
            v    = int(255 * fade)
            cv2.putText(frame, f"{note:<6} {age:.1f}s", (dx, dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, v, int(v*0.5)), 1)
            dy += 14
            if dy > by + BOX_H - 8: break


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Air Piano")
    print("=" * 60)
    print(f"  Audio : {'sounddevice (ON)' if AUDIO else 'DISABLED — pip install sounddevice'}")
    print()
    print("  PHASES  (all automatic — just use your hands)")
    print("  1. SETUP      — step back until face + both hands visible")
    print("  2. CALIBRATE  — hover hands at playing height for 4 s")
    print("  3. TRAINING   — press each key shown (random order, 3 presses each)")
    print("  4. PLAYING    — play freely")
    print()
    print("  R — restart    Q — quit")
    print()
    AirPiano().run()


if __name__ == "__main__":
    main()
