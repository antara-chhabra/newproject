#!/usr/bin/env python3
"""
Air Piano — Play piano in the air using your webcam.

Two-tier height model:
  White keys — played at your natural resting hand height   (cyan baseline)
  Black keys — played from a raised position, as if reaching for an elevated
               key further back on a real piano              (purple baseline)

Phases (fully automatic):
  SETUP        — step back until face + both hands visible
  CALIBRATE    — measure white key baseline (resting height for 4 s)
  TRAIN_WHITE  — 5 presses per white key, random order, no timer escape
  TRAIN_BLACK  — 7 presses per black key, random order, finger raised
  PLAYING      — personalized two-tier key detection

Profile saved to profile.json after training. Auto-loaded on next launch.

Controls:  R = restart   Q = quit
"""

import cv2
import json
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
from datetime import datetime
from typing import Optional

# ── Model ─────────────────────────────────────────────────────────────────────
_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
_MODEL_URL   = ("https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
PROFILE_PATH = os.path.join(os.path.dirname(__file__), "profile.json")

def _ensure_model() -> str:
    if not os.path.exists(_MODEL_PATH):
        print("[Air Piano] Downloading model (~2 MB)…")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH

# ── Audio ─────────────────────────────────────────────────────────────────────
try:
    import sounddevice as sd
    AUDIO = True
except ImportError:
    AUDIO = False
    print("[!] sounddevice not found — audio disabled.")

SAMPLE_RATE = 44100

NOTES: dict[str, float] = {
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13,
    "E4": 329.63, "F4": 349.23, "F#4": 369.99, "G4": 392.00,
    "G#4": 415.30, "A4": 440.00, "A#4": 466.16, "B4": 493.88,
    "C5": 523.25,
}

WHITE    = ["C4",  "D4",  "E4",  "F4",  "G4",  "A4",  "B4",  "C5"]
BLACK    = ["C#4", "D#4", "F#4", "G#4", "A#4"]
BLACK_POS: dict[str, float] = {
    "C#4": 0.65, "D#4": 1.65, "F#4": 3.65, "G#4": 4.65, "A#4": 5.65,
}
# Nearest white key to fall back to when a finger is in a black zone at wrong tier
BLACK_TO_WHITE = {"C#4": "C4", "D#4": "D4", "F#4": "F4", "G#4": "G4", "A#4": "A4"}

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
    hsv = np.uint8([[[int(i / n * 145), 180, 200]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

ZONE_COLS = [_zone_col(i, len(WHITE)) for i in range(len(WHITE))]

# ── Training constants ────────────────────────────────────────────────────────
SETUP_HANDS_SECS   = 1.5
CALIBRATION_SECS   = 4
MIN_PRESSES_WHITE  = 5
MIN_PRESSES_BLACK  = 7
REQUEUE_TIMEOUT    = 10    # re-queue key if 0 presses after this many seconds
PRESS_COOLDOWN     = 0.32


# ── Audio engine ──────────────────────────────────────────────────────────────

def _synthesize(freq: float, dur: float = 0.75) -> np.ndarray:
    n = int(SAMPLE_RATE * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    wave = (np.sin(2*np.pi*freq*t)     * 0.50 +
            np.sin(2*np.pi*freq*2*t)   * 0.25 +
            np.sin(2*np.pi*freq*3*t)   * 0.12 +
            np.sin(2*np.pi*freq*0.5*t) * 0.08)
    a = max(1, int(0.005 * SAMPLE_RATE))
    d = max(1, int(0.040 * SAMPLE_RATE))
    r = max(1, int(0.200 * SAMPLE_RATE))
    env = np.ones(n) * 0.70
    env[:a]    = np.linspace(0,   1.0, a)
    env[a:a+d] = np.linspace(1.0, 0.7, d)
    env[-r:]   = np.linspace(0.7, 0.0, r)
    return (wave * env * 0.45).astype(np.float32)


class AudioEngine:
    """Callback-based mixer — multiple simultaneous notes, no thread conflicts."""

    def __init__(self):
        self._cache:  dict[str, np.ndarray] = {}
        self._voices: list[list]            = []
        self._lock    = threading.Lock()
        self._stream: Optional["sd.OutputStream"] = None
        if AUDIO:
            self._stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=1,
                dtype="float32", callback=self._callback, blocksize=512,
            )
            self._stream.start()

    def _callback(self, outdata: np.ndarray, frames: int, ti, st):
        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            rem = []
            for v in self._voices:
                data, pos = v
                end = min(pos + frames, len(data))
                out[:end - pos] += data[pos:end]
                if end < len(data):
                    v[1] = end
                    rem.append(v)
            self._voices = rem
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
    def __init__(self, x: int, y: int, color=(0, 210, 255)):
        self.x, self.y, self.color = x, y, color
        self._t0 = time.time()

    @property
    def alive(self) -> bool:
        return (time.time() - self._t0) < 0.65

    def draw(self, frame: np.ndarray):
        dt    = time.time() - self._t0
        alpha = max(0.0, 1.0 - dt / 0.65)
        ov    = frame.copy()
        cv2.circle(ov, (self.x, self.y), int(8 + dt * 110), self.color, 3)
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
        """Return key at px — black keys take priority over white."""
        rel = (px - self.ox) / self.tw
        if not (0.0 <= rel <= 1.0):
            return None
        wn = 1.0 / len(WHITE)
        for note, pos in BLACK_POS.items():
            b0 = pos * wn + (wn - wn * 0.56) / 2
            if b0 <= rel <= b0 + wn * 0.56:
                return note
        return WHITE[min(int(rel * len(WHITE)), len(WHITE) - 1)]

    def nearest_white_for_x(self, px: float) -> str:
        rel = max(0.0, min(1.0, (px - self.ox) / self.tw))
        return WHITE[min(int(rel * len(WHITE)), len(WHITE) - 1)]

    def draw(self, frame: np.ndarray, active: set[str],
             ripples: list[Ripple], highlight: Optional[str] = None):
        for i, note in enumerate(WHITE):
            x1, y1, x2, y2 = self.white_rect(i)
            col = ((110, 235, 140) if note in active else
                   (0, 220, 255)   if note == highlight else
                   (238, 238, 238))
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)
            cv2.putText(frame, note[:-1],
                        (x1 + int(self.wkw / 2) - 7, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)
        for note in BLACK:
            x1, y1, x2, y2 = self.black_rect(note)
            col = ((60, 210,  90) if note in active else
                   (0,  180, 200) if note == highlight else
                   (28,  28,  28))
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (110, 110, 110), 1)
        for r in ripples:
            r.draw(frame)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _centered_text(frame: np.ndarray, text: str, y: int, w: int,
                   scale: float, color, thickness: int = 1):
    tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    cv2.putText(frame, text, ((w - tw) // 2, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


# ── Main application ──────────────────────────────────────────────────────────

class AirPiano:
    PROFILE_LOAD = -1
    SETUP        =  0
    CALIBRATE    =  1
    TRAIN_WHITE  =  2
    TRAIN_BLACK  =  3
    PLAYING      =  4

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
        self._init_state()

    def _init_state(self):
        self.phase             = self.SETUP
        self.phase_start       = time.time()
        # SETUP
        self._setup_both_since: Optional[float] = None
        # CALIBRATE
        # Collects y-pixel of every fingertip each frame for CALIBRATION_SECS
        # seconds.  70th-percentile → white_baseline_y.
        self.cal_ys:           list[float] = []
        self.white_baseline_y: float       = 0.0
        # TRAINING shared state
        self.training_queue:   list[str]   = []   # keys to train this round
        self.training_missed:  list[str]   = []   # keys with 0 presses → re-queued
        self.training_key_idx: int         = 0
        self.training_presses: int         = 0
        self.key_start_time:   float       = 0.0
        self.finger_down:      dict[str, bool]         = {}
        # Raw training data
        # train_depths[(hand, key)]     = list of press-depth px values
        # train_resting_ys[(hand, key)] = list of y-px values sampled while
        #                                  hovering above white baseline (black keys only)
        self.train_depths:     dict[tuple[str,str], list[float]] = {}
        self.train_resting_ys: dict[tuple[str,str], list[float]] = {}
        # Computed after training
        self.press_thresholds: dict[tuple[str,str], float] = {}
        self.black_resting_y:  dict[tuple[str,str], float] = {}
        self.black_baseline_y: float = 0.0
        self.tier_boundary:    float = 0.0   # midpoint; py < this → elevated tier
        # PLAYING
        self.active:   set[str]                  = set()
        self.cooldown: dict[str, float]          = {}
        self.ripples:  list[Ripple]              = []
        self.history:  deque[tuple[float, str]]  = deque(maxlen=30)
        # PROFILE_LOAD display timer
        self._profile_loaded_until: float = 0.0

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.audio.preload()
        profile_checked = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # Try loading profile once we know the frame size
            if not profile_checked:
                profile_checked = True
                if self._load_profile(h):
                    self.phase = self.PROFILE_LOAD
                    self._profile_loaded_until = time.time() + 2.5

            piano_y = int(h * 0.72)
            layout  = PianoLayout(0, piano_y, w, h - piano_y)

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res    = self.landmarker.detect_for_video(mp_img, int(time.time() * 1000))

            tips: list[tuple[int, int, str, int, str]] = []
            if res.hand_landmarks and res.handedness:
                for hidx, hand_lm in enumerate(res.hand_landmarks):
                    hl = res.handedness[hidx][0].category_name
                    for i, tid in enumerate(TIPS):
                        lm = hand_lm[tid]
                        tips.append((int(lm.x * w), int(lm.y * h),
                                     TIP_NAMES[i], i, hl))

            if   self.phase == self.PROFILE_LOAD: self._phase_profile_load(frame, h, w)
            elif self.phase == self.SETUP:        self._phase_setup(frame, tips, h, w)
            elif self.phase == self.CALIBRATE:    self._phase_calibrate(frame, tips, h, w)
            elif self.phase == self.TRAIN_WHITE:  self._phase_train(frame, tips, h, w, layout, "white")
            elif self.phase == self.TRAIN_BLACK:  self._phase_train(frame, tips, h, w, layout, "black")
            else:                                 self._phase_playing(frame, tips, h, w, layout)

            cv2.imshow("Air Piano", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("r"):
                self._init_state()
                print("[Air Piano] Reset.")

        self.audio.close()
        self.landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

    # ── Profile ───────────────────────────────────────────────────────────────

    def _save_profile(self, frame_height: int):
        def ser(d: dict) -> dict:
            return {f"{k[0]}|{k[1]}": v for k, v in d.items()}
        data = {
            "white_baseline_y": self.white_baseline_y,
            "black_baseline_y": self.black_baseline_y,
            "tier_boundary":    self.tier_boundary,
            "press_thresholds": ser(self.press_thresholds),
            "black_resting_y":  ser(self.black_resting_y),
            "frame_height":     frame_height,
            "trained_at":       datetime.now().isoformat(timespec="seconds"),
        }
        with open(PROFILE_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Air Piano] Profile saved → {PROFILE_PATH}")

    def _load_profile(self, frame_height: int) -> bool:
        if not os.path.exists(PROFILE_PATH):
            return False
        try:
            with open(PROFILE_PATH) as f:
                p = json.load(f)
            if p.get("frame_height") != frame_height:
                print("[Air Piano] Profile resolution mismatch — needs retrain.")
                return False
            def deser(d: dict) -> dict:
                return {tuple(k.split("|")): v for k, v in d.items()}
            self.white_baseline_y = p["white_baseline_y"]
            self.black_baseline_y = p["black_baseline_y"]
            self.tier_boundary    = p["tier_boundary"]
            self.press_thresholds = deser(p["press_thresholds"])
            self.black_resting_y  = deser(p["black_resting_y"])
            print(f"[Air Piano] Profile loaded (trained {p.get('trained_at', '?')}).")
            return True
        except Exception as e:
            print(f"[Air Piano] Profile load failed: {e}")
            return False

    def _phase_profile_load(self, frame: np.ndarray, h: int, w: int):
        if time.time() >= self._profile_loaded_until:
            self.phase = self.PLAYING
            return
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        _centered_text(frame, "Profile loaded — starting play mode",
                        h // 2 - 20, w, 0.75, (0, 220, 130), 1)
        _centered_text(frame, "Press  R  to retrain from scratch",
                        h // 2 + 20, w, 0.45, (140, 140, 160), 1)

    # ── Phase: SETUP ──────────────────────────────────────────────────────────

    def _phase_setup(self, frame: np.ndarray, tips, h: int, w: int):
        both = len(tips) >= 8
        if both:
            if self._setup_both_since is None:
                self._setup_both_since = time.time()
            held = time.time() - self._setup_both_since
            if held >= SETUP_HANDS_SECS:
                self.phase       = self.CALIBRATE
                self.phase_start = time.time()
                self.cal_ys.clear()
                return
        else:
            self._setup_both_since = None
            held = 0.0

        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.50, frame, 0.50, 0, frame)

        y0 = h // 7
        for i, (txt, sz, col, th) in enumerate([
            ("AIR  PIANO",                       1.05, (0, 225, 130),   2),
            ("",                                  0.0,  (0, 0, 0),      0),
            ("Step back until your face",         0.55, (175, 175, 175), 1),
            ("and both hands are clearly",        0.55, (175, 175, 175), 1),
            ("visible in the frame.",             0.55, (175, 175, 175), 1),
        ]):
            if not txt or sz == 0.0: continue
            _centered_text(frame, txt, y0 + i * 50, w, sz, col, th)

        if both:
            prog = min(1.0, held / SETUP_HANDS_SECS)
            _centered_text(frame, "Both hands detected — hold still…",
                           h // 2 + 55, w, 0.65, (0, 255, 120), 1)
            bw = int(w * 0.35); bx = (w - bw) // 2; by = h // 2 + 80
            cv2.rectangle(frame, (bx, by), (bx + bw, by + 14), (45, 45, 55), -1)
            cv2.rectangle(frame, (bx, by), (bx + int(bw * prog), by + 14), (0, 215, 120), -1)
        else:
            _centered_text(frame, f"Hands visible: {len(tips)//5} / 2",
                           h // 2 + 55, w, 0.65, (0, 165, 255), 1)

        for px, py, _, cidx, _ in tips:
            cv2.circle(frame, (px, py), 10, TIP_COLS[cidx], -1)
            cv2.circle(frame, (px, py), 12, (255, 255, 255), 2)

        cv2.putText(frame, "R: restart    Q: quit",
                    (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 110), 1)

    # ── Phase: CALIBRATE ──────────────────────────────────────────────────────

    def _phase_calibrate(self, frame: np.ndarray, tips, h: int, w: int):
        """
        Sample the y-pixel of every fingertip for CALIBRATION_SECS seconds.
        70th percentile → white_baseline_y (resting hand height for white keys).
        Auto-advances when timer expires.
        """
        elapsed = time.time() - self.phase_start
        if elapsed >= CALIBRATION_SECS:
            self.white_baseline_y = (float(np.percentile(self.cal_ys, 70))
                                     if self.cal_ys else h * 0.60)
            print(f"[Air Piano] White baseline y = {self.white_baseline_y:.1f}")
            self._start_train("white")
            return

        prog = elapsed / CALIBRATION_SECS
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)

        y0 = h // 7
        for i, (txt, sz, col, th) in enumerate([
            ("CALIBRATING",                          0.75, (200, 200, 200), 1),
            ("",                                      0.0,  (0, 0, 0),     0),
            ("Hold hands at your natural",            0.50, (175, 175, 175), 1),
            ("playing height and stay still.",        0.50, (175, 175, 175), 1),
            ("",                                      0.0,  (0, 0, 0),     0),
            ("Measuring your hand height…",           0.50, (0, 205, 255),  1),
            (f"{max(0.0, CALIBRATION_SECS - elapsed):.1f}s", 0.45, (140, 140, 170), 1),
        ]):
            if not txt or sz == 0.0: continue
            _centered_text(frame, txt, y0 + i * 46, w, sz, col, th)

        bw = int(w * 0.50); bx = (w - bw) // 2; by = h // 2 + 50
        cv2.rectangle(frame, (bx, by), (bx + bw, by + 18), (45, 45, 55), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(bw * prog), by + 18), (0, 215, 120), -1)

        for px, py, _, cidx, _ in tips:
            self.cal_ys.append(float(py))
            cv2.circle(frame, (px, py), 10, TIP_COLS[cidx], -1)
            cv2.circle(frame, (px, py), 12, (255, 255, 255), 2)

    # ── Phase: TRAINING ───────────────────────────────────────────────────────

    def _start_train(self, tier: str):
        keys = WHITE if tier == "white" else BLACK
        self.training_queue   = random.sample(keys, len(keys))
        self.training_missed  = []
        self.training_key_idx = 0
        self.training_presses = 0
        self.finger_down.clear()
        self.key_start_time   = time.time()
        self.phase            = self.TRAIN_WHITE if tier == "white" else self.TRAIN_BLACK
        print(f"[Air Piano] Training {tier} keys: {self.training_queue}")

    def _phase_train(self, frame: np.ndarray, tips, h: int, w: int,
                     layout: PianoLayout, tier: str):
        # ── Check if current queue is exhausted ───────────────────────────
        if self.training_key_idx >= len(self.training_queue):
            if self.training_missed:
                # Re-run missed keys
                self.training_queue   = self.training_missed[:]
                self.training_missed  = []
                self.training_key_idx = 0
                self.training_presses = 0
                self.finger_down.clear()
                self.key_start_time   = time.time()
                print(f"[Air Piano] Re-queuing missed {tier} keys: {self.training_queue}")
            else:
                self._finish_train(tier, h)
                return

        target_key = self.training_queue[self.training_key_idx]
        min_presses = MIN_PRESSES_WHITE if tier == "white" else MIN_PRESSES_BLACK
        elapsed_key = time.time() - self.key_start_time

        # ── Re-queue if ignored too long ──────────────────────────────────
        if elapsed_key >= REQUEUE_TIMEOUT and self.training_presses == 0:
            print(f"[Air Piano] '{target_key}' not pressed — re-queuing.")
            self.training_missed.append(target_key)
            self._next_training_key()
            return

        # ── Advance when enough presses collected ─────────────────────────
        if self.training_presses >= min_presses:
            self._next_training_key()
            return

        bl = int(self.white_baseline_y)

        # Zone bands
        ov = frame.copy()
        zone_top = max(0, bl - 5)
        if layout.oy > zone_top:
            for i, note in enumerate(WHITE):
                x1, _, x2, _ = layout.white_rect(i)
                cv2.rectangle(ov, (x1, zone_top), (x2, layout.oy), ZONE_COLS[i], -1)
            cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)

        # White baseline (cyan)
        cv2.line(frame, (0, bl), (w, bl), (0, 235, 200), 2)
        _centered_text(frame, "white key level", bl - 8, w, 0.30, (0, 200, 170), 1)

        # Black baseline (purple) if we have samples
        if tier == "black":
            all_resting = [y for ys in self.train_resting_ys.values() for y in ys]
            if len(all_resting) > 10:
                running_bbl = float(np.median(all_resting))
                cv2.line(frame, (0, int(running_bbl)), (w, int(running_bbl)),
                         (200, 80, 200), 2)
                _centered_text(frame, "black key level",
                               int(running_bbl) - 8, w, 0.30, (180, 80, 180), 1)

        layout.draw(frame, set(), [], highlight=target_key)

        # Dim upper area for text
        ov2 = frame.copy()
        cv2.rectangle(ov2, (0, 0), (w, int(h * 0.60)), (0, 0, 0), -1)
        cv2.addWeighted(ov2, 0.38, frame, 0.62, 0, frame)

        round_str = "ROUND 1 — WHITE KEYS" if tier == "white" else "ROUND 2 — BLACK KEYS"
        instruction = ("Press naturally at your resting height"
                       if tier == "white" else
                       "Raise your finger slightly — reach for an elevated key")

        y0 = h // 10
        for i, (txt, sz, col, th) in enumerate([
            ("TRAINING",                                          0.60, (200, 200, 200), 1),
            (round_str,                                           0.48, (150, 150, 200), 1),
            ("",                                                   0.0,  (0,0,0), 0),
            (f"Press   {target_key}",                             1.10, (0, 220, 200), 2),
            ("",                                                   0.0,  (0,0,0), 0),
            (instruction,                                          0.42, (160, 200, 160), 1),
            ("",                                                   0.0,  (0,0,0), 0),
            (f"Key {self.training_key_idx+1} / {len(self.training_queue)}"
             f"   •   {self.training_presses} / {min_presses} presses",
                                                                   0.48, (150,150,180), 1),
        ]):
            if not txt or sz == 0.0: continue
            _centered_text(frame, txt, y0 + i * 48, w, sz, col, th)

        # Countdown bar (only visible while waiting for first press)
        if self.training_presses == 0:
            prog = min(1.0, elapsed_key / REQUEUE_TIMEOUT)
            bw = int(w * 0.40); bx = (w - bw) // 2; by = int(h * 0.60) + 8
            cv2.rectangle(frame, (bx, by), (bx + bw, by + 10), (45, 45, 55), -1)
            cv2.rectangle(frame, (bx, by), (bx + int(bw * prog), by + 10),
                          (80, 80, 180), -1)
            _centered_text(frame,
                           f"re-queuing in {max(0, REQUEUE_TIMEOUT - elapsed_key):.0f}s "
                           f"if not pressed",
                           by + 26, w, 0.32, (100, 100, 140), 1)

        # ── Detect presses ────────────────────────────────────────────────
        for px, py, fname, cidx, hand in tips:
            key  = layout.key_for_x(px)
            fkey = f"{hand}_{fname}"

            if tier == "white":
                depth       = py - self.white_baseline_y
                pressed_now = key == target_key and depth >= 10
                was_down    = self.finger_down.get(fkey, False)
                if pressed_now and not was_down:
                    self.training_presses += 1
                    self.train_depths.setdefault((hand, target_key), []).append(float(depth))
                    self.audio.play(target_key)
                self.finger_down[fkey] = pressed_now

            else:  # black key training
                if key == target_key:
                    # Sample resting y when clearly above white baseline (elevated tier)
                    if py < self.white_baseline_y - 8:
                        self.train_resting_ys.setdefault(
                            (hand, target_key), []).append(float(py))

                    # Compute running resting y for this (hand, key)
                    resting_samples = self.train_resting_ys.get((hand, target_key), [])
                    if resting_samples:
                        resting_y = float(np.median(resting_samples))
                    else:
                        resting_y = self.white_baseline_y - 25  # fallback estimate

                    depth       = py - resting_y
                    pressed_now = depth >= 12
                    was_down    = self.finger_down.get(fkey, False)
                    if pressed_now and not was_down:
                        self.training_presses += 1
                        self.train_depths.setdefault((hand, target_key), []).append(float(depth))
                        self.audio.play(target_key)
                    self.finger_down[fkey] = pressed_now
                else:
                    self.finger_down[f"{hand}_{fname}"] = False

            # Draw fingertip
            cv2.circle(frame, (px, py), 11, TIP_COLS[cidx], -1)
            cv2.circle(frame, (px, py), 13, (255, 255, 255), 2)

            # Depth indicator bar
            if tier == "white":
                ref_y = self.white_baseline_y
            else:
                resting_samples = self.train_resting_ys.get((hand, key or ""), [])
                ref_y = float(np.median(resting_samples)) if resting_samples else self.white_baseline_y - 25
            depth_vis = py - ref_y
            if 0 < depth_vis < 80 and key == target_key:
                intensity = min(1.0, depth_vis / 40)
                cv2.line(frame, (px, int(ref_y)), (px, py),
                         tuple(int(c * intensity) for c in TIP_COLS[cidx]), 4)

    def _next_training_key(self):
        self.training_key_idx += 1
        self.training_presses  = 0
        self.finger_down.clear()
        self.key_start_time    = time.time()

    def _finish_train(self, tier: str, h: int):
        if tier == "white":
            print("[Air Piano] White key training complete.")
            self._start_train("black")
        else:
            print("[Air Piano] Black key training complete. Computing thresholds…")
            self._compute_thresholds(h)
            self.phase = self.PLAYING
            print("[Air Piano] Ready to play!")

    def _compute_thresholds(self, h: int):
        """
        Press thresholds:
          For each (hand, key), take the 40th-percentile of recorded press depths.
          This captures a light but intentional press rather than maximum depth.
          Pairs with no data fall back to 20 px.

        Black key resting heights:
          For each (hand, key), take the median of resting y samples collected
          while the user hovered above the white baseline during black key training.
          This is the elevated tier position for that finger on that key.

        black_baseline_y:
          Overall median of all black resting y values → used for the purple line
          and as a fallback for (hand, key) pairs with no black resting data.

        tier_boundary:
          Midpoint between white_baseline_y and black_baseline_y.
          During play: py < tier_boundary → elevated tier → black key intent.
        """
        # Press thresholds for all keys
        for (hand, key), depths in self.train_depths.items():
            thresh = max(8.0, min(60.0, float(np.percentile(depths, 40))))
            self.press_thresholds[(hand, key)] = thresh
            print(f"  threshold  {hand:<6} {key:<5} = {thresh:.1f} px")

        # Black resting heights
        all_resting_y = []
        for (hand, key), ys in self.train_resting_ys.items():
            if ys:
                rv = float(np.median(ys))
                self.black_resting_y[(hand, key)] = rv
                all_resting_y.extend(ys)
                print(f"  black_rest {hand:<6} {key:<5} = {rv:.1f} px")

        if all_resting_y:
            self.black_baseline_y = float(np.median(all_resting_y))
        else:
            self.black_baseline_y = self.white_baseline_y - 30  # fallback

        self.tier_boundary = (self.white_baseline_y + self.black_baseline_y) / 2
        print(f"  white baseline  = {self.white_baseline_y:.1f}")
        print(f"  black baseline  = {self.black_baseline_y:.1f}")
        print(f"  tier boundary   = {self.tier_boundary:.1f}")

        # Save profile using frame height encoded in white_baseline (rough proxy)
        # Actual frame height saved at run() level below
        self._save_pending = True

    # ── Phase: PLAYING ────────────────────────────────────────────────────────

    def _phase_playing(self, frame: np.ndarray, tips, h: int, w: int,
                       layout: PianoLayout):
        # Save profile on first play frame (we now know h)
        if getattr(self, "_save_pending", False):
            self._save_profile(h)
            self._save_pending = False

        now = time.time()
        bl  = int(self.white_baseline_y)
        bbl = int(self.black_baseline_y)
        new_active: set[str] = set()

        # Zone bands
        ov = frame.copy()
        zone_top = max(0, bbl - 5)
        if layout.oy > zone_top:
            for i, note in enumerate(WHITE):
                x1, _, x2, _ = layout.white_rect(i)
                cv2.rectangle(ov, (x1, zone_top), (x2, layout.oy), ZONE_COLS[i], -1)
            cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)

        # White baseline (cyan) and black baseline (purple)
        cv2.line(frame, (0, bl), (w, bl), (0, 235, 200), 2)
        cv2.putText(frame, "white keys", (8, bl - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 210, 170), 1)
        if bbl > 0 and bbl < bl:
            cv2.line(frame, (0, bbl), (w, bbl), (200, 80, 200), 2)
            cv2.putText(frame, "black keys", (8, bbl - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 80, 180), 1)

        for px, py, fname, cidx, hand in tips:
            col = TIP_COLS[cidx]

            # ── Two-tier key resolution ───────────────────────────────────
            x_key = layout.key_for_x(px)
            if x_key in BLACK and self.tier_boundary > 0:
                if py < self.tier_boundary:
                    key  = x_key          # elevated tier → black key intent
                    tier = "black"
                else:
                    key  = layout.nearest_white_for_x(px)  # white tier → white key
                    tier = "white"
            else:
                key  = x_key
                tier = "white"

            # ── Press detection ───────────────────────────────────────────
            if tier == "white":
                ref_y  = self.white_baseline_y
                thresh = self.press_thresholds.get((hand, key), 20)
            else:
                ref_y  = self.black_resting_y.get((hand, key), self.black_baseline_y)
                thresh = self.press_thresholds.get((hand, key), 15)

            depth       = py - ref_y
            fkey        = f"{hand}_{fname}"
            pressed_now = depth >= thresh
            was_down    = self.finger_down.get(fkey, False)

            if pressed_now and key:
                new_active.add(key)
                if not was_down:
                    cd = self.cooldown.get(key, 0.0)
                    if now - cd > PRESS_COOLDOWN:
                        self.cooldown[key] = now
                        self.audio.play(key)
                        self.history.appendleft((now, key))
                        self.ripples.append(Ripple(px, int(ref_y), col))

            self.finger_down[fkey] = pressed_now

            # Fingertip dot
            cv2.circle(frame, (px, py), 11, col, -1)
            cv2.circle(frame, (px, py), 13, (255, 255, 255), 2)

            # Depth bar (from ref_y down to fingertip)
            if 0 < depth < thresh * 3 and key:
                intensity = min(1.0, depth / thresh)
                cv2.line(frame, (px, int(ref_y)), (px, py),
                         tuple(int(c * intensity) for c in col), 4)
                ov2 = frame.copy()
                cv2.circle(ov2, (px, py), int(11 + intensity * 14), col, 2)
                cv2.addWeighted(ov2, intensity * 0.7, frame,
                                1 - intensity * 0.7, 0, frame)

            # Small tier label above fingertip
            if self.tier_boundary > 0:
                label_col = (200, 80, 200) if tier == "black" else (0, 210, 170)
                cv2.putText(frame, "B" if tier == "black" else "W",
                            (px - 4, py - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, label_col, 1)

        self.active  = new_active
        self.ripples = [r for r in self.ripples if r.alive]
        layout.draw(frame, self.active, self.ripples)
        self._draw_stats(frame, h, w)

        cv2.putText(frame, "R: restart    Q: quit",
                    (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 100), 1)

    # ── Stats overlay (top-right) ─────────────────────────────────────────────

    def _draw_stats(self, frame: np.ndarray, h: int, w: int):
        BOX_W, BOX_H = 210, 180
        bx, by = w - BOX_W - 10, 10

        ov = frame.copy()
        cv2.rectangle(ov, (bx, by), (bx + BOX_W, by + BOX_H), (12, 15, 22), -1)
        cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (bx, by), (bx + BOX_W, by + BOX_H), (48, 52, 68), 1)

        dx, dy = bx + 8, by + 18
        cv2.putText(frame, "AIR PIANO", (dx, dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 130), 1)
        dy += 6
        cv2.line(frame, (bx + 4, dy), (bx + BOX_W - 4, dy), (48, 52, 68), 1)
        dy += 15

        cv2.putText(frame, "NOW PLAYING", (dx, dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (110, 110, 145), 1)
        dy += 15
        act = "  ".join(sorted(self.active)) if self.active else u"—"
        cv2.putText(frame, act[:18], (dx, dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)
        dy += 17
        cv2.line(frame, (bx + 4, dy - 2), (bx + BOX_W - 4, dy - 2), (48, 52, 68), 1)

        cv2.putText(frame, "RECENT", (dx, dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (110, 110, 145), 1)
        dy += 15
        now = time.time()
        for ts, note in list(self.history)[:6]:
            age  = now - ts
            if age > 8: continue
            fade = max(0.2, 1.0 - age / 6.0)
            v    = int(255 * fade)
            cv2.putText(frame, f"{note:<6} {age:.1f}s", (dx, dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, v, int(v * 0.5)), 1)
            dy += 14
            if dy > by + BOX_H - 8: break


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Air Piano — Two-Tier Height Model")
    print("=" * 60)
    print(f"  Audio : {'sounddevice (ON)' if AUDIO else 'DISABLED'}")
    print()
    print("  PHASES  (automatic — use hands only)")
    print("  SETUP        step back, show face + both hands")
    print("  CALIBRATE    hover at playing height for 4 s")
    print("  TRAIN WHITE  press each white key shown (5× each, random order)")
    print("  TRAIN BLACK  raise finger, press each black key (7× each)")
    print("  PLAYING      two-tier detection active")
    print()
    print("  Cyan line   = white key baseline")
    print("  Purple line = black key baseline  (B/W label on each fingertip)")
    print()
    print("  R restart   Q quit")
    print()
    AirPiano().run()


if __name__ == "__main__":
    main()
