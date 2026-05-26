#!/usr/bin/env python3
"""
Air Piano — Play piano in the air using your webcam.

Controls:
  Q  — quit
  R  — re-run calibration
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from collections import deque
from typing import Optional

try:
    import sounddevice as sd
    AUDIO = True
except ImportError:
    AUDIO = False
    print("[!] sounddevice not found — audio disabled.  pip install sounddevice")

SAMPLE_RATE       = 44100
CALIBRATION_SECS  = 4       # how long calibration lasts
PRESS_THRESHOLD   = 22      # px below baseline = key press
PRESS_COOLDOWN    = 0.32    # s before same key can trigger again

# ── Note frequencies (Hz) ─────────────────────────────────────────────────────
NOTES: dict[str, float] = {
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13,
    "E4": 329.63, "F4": 349.23, "F#4": 369.99, "G4": 392.00,
    "G#4": 415.30, "A4": 440.00, "A#4": 466.16, "B4": 493.88,
    "C5": 523.25,
}

# One octave: 8 white keys, 5 black keys
WHITE = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
BLACK = ["C#4", "D#4", "F#4", "G#4", "A#4"]

# Black key position as fraction from left (in white-key units)
BLACK_POS: dict[str, float] = {
    "C#4": 0.65, "D#4": 1.65, "F#4": 3.65, "G#4": 4.65, "A#4": 5.65,
}

# MediaPipe fingertip landmark IDs
TIPS      = [4, 8, 12, 16, 20]
TIP_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
TIP_COLS  = [           # BGR
    (0,   165, 255),    # orange  — Thumb
    (80,   80, 255),    # red     — Index
    (80,  255,  80),    # green   — Middle
    (255,  80,  80),    # blue    — Ring
    (200,   0, 200),    # purple  — Pinky
]

# Zone colour per white key (rainbow gradient)
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
    """Pixel geometry for white + black keys within a bounding rect."""

    def __init__(self, x: int, y: int, w: int, h: int):
        self.ox, self.oy, self.tw, self.th = x, y, w, h
        self.wkw  = w / len(WHITE)          # white key width (px)
        self.bkw  = self.wkw * 0.56         # black key width (px)
        self.bkh  = h * 0.62               # black key height (px)

    # ── Geometry helpers ──────────────────────────────────────────────────────

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
        """Map an x pixel to the note whose zone contains it (black takes priority)."""
        rel = (px - self.ox) / self.tw
        if not (0.0 <= rel <= 1.0):
            return None
        n    = len(WHITE)
        wn   = 1.0 / n          # normalised white key width
        bwn  = wn * 0.56
        for note, pos in BLACK_POS.items():
            b0 = pos * wn + (wn - bwn) / 2
            b1 = b0 + bwn
            if b0 <= rel <= b1:
                return note
        return WHITE[min(int(rel * n), n - 1)]

    # ── Drawing ───────────────────────────────────────────────────────────────

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


# ── Main application ──────────────────────────────────────────────────────────

class AirPiano:
    CALIBRATING = 0
    PLAYING     = 1

    def __init__(self):
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.72,
            min_tracking_confidence=0.60,
        )
        self.audio = AudioEngine()

        self.phase      = self.CALIBRATING
        self.cal_start  = time.time()
        self.cal_ys: list[float] = []
        self.baseline_y = 0.0

        self.active:   set[str]                      = set()
        self.cooldown: dict[str, float]              = {}
        self.finger_down: dict[str, bool]            = {}
        self.ripples:  list[Ripple]                  = []
        self.history:  deque[tuple[float, str]]      = deque(maxlen=30)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("[Air Piano] Pre-caching audio…")
        self.audio.preload()
        print("[Air Piano] Ready.  Hold hands in playing position for calibration.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            DASH_W = 225
            CAM_W  = w - DASH_W

            piano_y = int(h * 0.79)
            piano_h = h - piano_y
            layout  = PianoLayout(0, piano_y, CAM_W, piano_h)

            # Hand detection on camera region only
            rgb = cv2.cvtColor(frame[:, :CAM_W], cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            # Collect fingertips: (px, py, finger_name, color_idx, hand_idx)
            tips: list[tuple[int, int, str, int, int]] = []
            if res.multi_hand_landmarks:
                for hidx, hand_lm in enumerate(res.multi_hand_landmarks):
                    for i, tid in enumerate(TIPS):
                        lm = hand_lm.landmark[tid]
                        px = int(lm.x * CAM_W)
                        py = int(lm.y * h)
                        tips.append((px, py, TIP_NAMES[i], i, hidx))

            # ── Calibration phase ─────────────────────────────────────────
            if self.phase == self.CALIBRATING:
                elapsed = time.time() - self.cal_start
                self._draw_calibration(frame, tips, h, CAM_W, elapsed)
                for px, py, *_ in tips:
                    self.cal_ys.append(float(py))
                if elapsed >= CALIBRATION_SECS:
                    if self.cal_ys:
                        # 70th percentile = typical "rest" finger height
                        self.baseline_y = float(np.percentile(self.cal_ys, 70))
                    else:
                        self.baseline_y = h * 0.58
                    print(f"[Air Piano] Baseline y = {self.baseline_y:.1f} px")
                    self.phase = self.PLAYING

            # ── Playing phase ─────────────────────────────────────────────
            else:
                now = time.time()
                new_active: set[str] = set()

                # Coloured zone bands between baseline and piano
                self._draw_zones(frame, layout, CAM_W)

                # Baseline line
                bl = int(self.baseline_y)
                cv2.line(frame, (0, bl), (CAM_W, bl), (0, 235, 200), 2)
                for i, note in enumerate(WHITE):
                    x1, _, x2, _ = layout.white_rect(i)
                    mid = (x1 + x2) // 2 - 6
                    cv2.putText(frame, note[:-1], (mid, bl - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 210, 170), 1)

                # Process each fingertip
                for px, py, fname, cidx, hidx in tips:
                    col  = TIP_COLS[cidx]
                    fkey = f"h{hidx}_{fname}"
                    key  = layout.key_for_x(px)

                    depth        = py - bl            # positive = below baseline
                    pressed_now  = depth >= PRESS_THRESHOLD
                    was_down     = self.finger_down.get(fkey, False)

                    if pressed_now and key:
                        new_active.add(key)
                        if not was_down:              # rising edge → new press
                            cd = self.cooldown.get(key, 0.0)
                            if now - cd > PRESS_COOLDOWN:
                                self.cooldown[key] = now
                                self.audio.play(key)
                                self.history.appendleft((now, key))
                                self.ripples.append(Ripple(px, bl, col))
                    self.finger_down[fkey] = pressed_now

                    # Draw fingertip dot
                    cv2.circle(frame, (px, py), 11, col, -1)
                    cv2.circle(frame, (px, py), 13, (255, 255, 255), 2)

                    # Press-depth bar (vertical line from baseline to tip)
                    if 0 < depth < PRESS_THRESHOLD * 3 and key:
                        intensity = min(1.0, depth / PRESS_THRESHOLD)
                        bar_col   = tuple(int(c * intensity) for c in col)
                        cv2.line(frame, (px, bl), (px, py), bar_col, 4)
                        # Glow ring that grows with depth
                        glow_r = int(11 + intensity * 14)
                        ov = frame.copy()
                        cv2.circle(ov, (px, py), glow_r, col, 2)
                        cv2.addWeighted(ov, intensity * 0.7, frame, 1 - intensity * 0.7, 0, frame)

                self.active  = new_active
                self.ripples = [r for r in self.ripples if r.alive]
                layout.draw(frame, self.active, self.ripples)
                self._draw_dashboard(frame, h, w, DASH_W, CAM_W)

            cv2.imshow("Air Piano", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("r"):
                self.phase      = self.CALIBRATING
                self.cal_start  = time.time()
                self.cal_ys.clear()
                self.active.clear()
                self.finger_down.clear()
                print("[Air Piano] Recalibrating…")

        cap.release()
        cv2.destroyAllWindows()

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_zones(self, frame: np.ndarray, layout: PianoLayout, cam_w: int):
        """Translucent rainbow columns showing each note's hover zone."""
        zone_top = max(0, int(self.baseline_y) - 5)
        zone_bot = layout.oy
        if zone_bot <= zone_top:
            return
        ov = frame.copy()
        for i, note in enumerate(WHITE):
            x1, _, x2, _ = layout.white_rect(i)
            cv2.rectangle(ov, (x1, zone_top), (x2, zone_bot), ZONE_COLS[i], -1)
        cv2.addWeighted(ov, 0.16, frame, 0.84, 0, frame)

    def _draw_calibration(self, frame: np.ndarray, tips, h: int, cam_w: int, elapsed: float):
        prog = min(1.0, elapsed / CALIBRATION_SECS)

        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (cam_w, h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)

        lines = [
            ("AIR  PIANO",                      1.05, (0, 225, 130), 2),
            ("",                                 0.0,  (0, 0, 0),    0),
            ("CALIBRATION",                      0.65, (200, 200, 200), 1),
            ("",                                 0.0,  (0, 0, 0),    0),
            ("Hold your hands in your natural",  0.50, (175, 175, 175), 1),
            ("playing position and keep still.", 0.50, (175, 175, 175), 1),
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

        # Detected fingertips
        for px, py, _, cidx, _h in tips:
            cv2.circle(frame, (px, py), 10, TIP_COLS[cidx], -1)
            cv2.circle(frame, (px, py), 12, (255, 255, 255), 2)

        cv2.putText(frame, "R: recalibrate    Q: quit",
                    (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 110), 1)

    def _draw_dashboard(self, frame: np.ndarray, h: int, w: int, dash_w: int, cam_w: int):
        # Background panel
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
        mk_w = dash_w - 14
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

        # Note history with fade bars
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
            bg_w = int((dash_w - 70) * fade)
            cv2.rectangle(frame, (dx, dy - 10), (dx + bg_w, dy - 2),
                          (0, int(v * 0.35), 0), -1)
            cv2.putText(frame, f"{note:<6} {age:.1f}s", (dx, dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, v, int(v * 0.5)), 1)
            dy += 17
            if dy > h - 45:
                break

        # Baseline info
        cv2.line(frame, (cam_w + 2, h - 38), (w - 2, h - 38), (48, 52, 68), 1)
        cv2.putText(frame, f"baseline y={int(self.baseline_y)}",
                    (dx, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 100), 1)
        cv2.putText(frame, "R recalibrate   Q quit",
                    (cam_w + 5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 80, 100), 1)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Air Piano")
    print("=" * 50)
    print(f"  Audio : {'sounddevice (ON)' if AUDIO else 'DISABLED (pip install sounddevice)'}")
    print("  Keys  : C4 D4 E4 F4 G4 A4 B4 C5 (+ sharps/flats)")
    print()
    print("  HOW TO PLAY")
    print("  1. Calibrate: hold hands in natural playing position")
    print("  2. The cyan line = your baseline (rest height)")
    print("  3. Press fingers DOWN past the baseline to play a note")
    print("  4. Your x-position selects the key (see colored zones)")
    print()
    AirPiano().run()


if __name__ == "__main__":
    main()
