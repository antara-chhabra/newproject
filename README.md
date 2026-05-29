# Air Piano

Play a piano with your bare hands in the air — no keyboard required. Uses your webcam, MediaPipe hand tracking, and OpenCV. Tracks both hands independently with personalized press calibration per finger per key.

---

## Setup

**Requirements:** Python 3.10+, a webcam.

```bash
cd newproject
pip install -r requirements.txt
python air_piano.py
```

On first run, the app automatically downloads a ~2 MB hand-landmarker model. After that it launches instantly.

---

## How to Play

The app runs through 4 phases — **fully automatic, no keyboard needed at any point.**

### Phase 1 — Setup
Step back from the screen until your face and both hands are clearly visible in the frame. Once both hands are detected for 1.5 continuous seconds, the app moves on automatically.

### Phase 2 — Calibration (4 seconds)
Hold both hands at the height you'd naturally hover above a piano. The app samples the y-position of every detected fingertip for 4 seconds, then takes the 70th percentile as your **baseline** — the pixel row representing your resting hand height. Any finger that dips below that line triggers a note.

### Phase 3 — Training (~1 minute)
The app presents all 13 keys in a **random order**. For each key:
- The target key is highlighted in cyan on the piano
- Press it naturally — the app records how deep your finger goes below the baseline
- After **3 confirmed presses** (or 4 seconds), the app moves to the next key automatically

The app learns a separate press threshold for each (hand, key) combination — so your left thumb on C4 and your right index on C4 can have different sensitivities.

### Phase 4 — Playing
Play freely. All 10 fingers work simultaneously. Your **left-right position** selects the key, your **downward motion** triggers it. Stats are shown in a small overlay box in the top-right corner.

---

## Keys

One octave: **C4 D4 E4 F4 G4 A4 B4 C5** (white keys) + **C#4 D#4 F#4 G#4 A#4** (black keys, narrower zones).

---

## Controls

| Key | Action |
|-----|--------|
| `R` | Restart from the beginning |
| `Q` | Quit |

No other keyboard input is needed. All phases advance automatically.

---

## How It Works

**Baseline measurement:** During calibration the app collects the y-pixel of every fingertip every frame. The 70th percentile of all those samples becomes `baseline_y`. This is stored as a single number (pixel row) for the session.

**Two-hand tracking:** MediaPipe reports handedness ("Left" / "Right") for each detected hand. Every fingertip tip is tagged with its hand label so left and right are handled independently throughout calibration, training, and play.

**Personalized thresholds:** Training depth samples are stored per `(hand, key)` pair. After training, the 40th-percentile depth for each pair becomes that combination's trigger threshold — capturing a light but intentional press. Pairs with no training data fall back to 20 px. These thresholds are stored in memory for the session only.

**Audio mixing:** Each note is pre-synthesized as a float32 numpy array with an ADSR envelope and harmonic overtones. A `sounddevice.OutputStream` runs a callback on a dedicated audio thread that mixes all currently playing voices into a single output buffer every 512 samples — this allows multiple simultaneous notes without interruption or crackling.

**Visual feedback:** A vertical depth bar grows from the baseline to each fingertip as you press down, brightening in the finger's colour as you approach the trigger threshold. A ripple effect fires at the baseline on each confirmed note.

---

## Tips

- **Good lighting** on your hands improves tracking significantly.
- Press **naturally and deliberately** during training — the personalization only works if your training presses reflect how you actually play.
- The **depth bar** is your guide: when it fully brightens, you're at your threshold.
- Press `R` to restart if you move closer or further from the camera.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No sound | `pip install sounddevice` |
| Webcam not opening | Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `air_piano.py` |
| Setup never advances | Improve lighting; step back further so both hands and face fit in frame |
| Notes fire too easily or not at all | Restart (`R`) and press more firmly or gently during training |
| One hand not tracked | Face the camera squarely; keep both hands fully in frame and well-lit |
