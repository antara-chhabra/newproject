# Air Piano

Play a piano with your bare hands in the air — no keyboard required. Uses your webcam, MediaPipe hand tracking, and OpenCV.

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

### Step 1 — Calibration (4 seconds)
When the app opens you'll see a calibration screen. Hold both hands out in front of you at the height you'd naturally hover above a piano — like you're about to play. Keep them still. The app records your resting finger height as your **baseline**.

### Step 2 — Playing
After calibration:
- A **cyan horizontal line** appears — that's your baseline.
- **Rainbow-coloured columns** show which note each zone plays (left = C4, right = C5).
- **Press a finger downward** past the cyan line to trigger a note.
- Your **left-right position** selects the key.
- All 10 fingers across both hands work simultaneously.

### Step 3 — Dashboard
The right panel shows:
- **Now Playing** — notes active this moment
- **Keyboard** — a mini piano that lights up green on each press
- **Recent Notes** — a fading log of everything played

---

## Keys

One octave: **C4 D4 E4 F4 G4 A4 B4 C5** (white keys) + **C#4 D#4 F#4 G#4 A#4** (black keys, narrower zones).

---

## Controls

| Key | Action |
|-----|--------|
| `R` | Re-run calibration |
| `Q` | Quit |

---

## Tips

- **Good lighting** on your hands makes tracking much more reliable.
- **Recalibrate** (`R`) any time you move closer or further from the camera.
- The **press depth bar** (vertical line from baseline to fingertip) glows brighter the deeper you press — use it to feel out the threshold before committing.
- Start **slow and deliberate**; once you have a feel for the depth, speed up.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No sound | `pip install sounddevice` |
| Webcam not opening | Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `air_piano.py` |
| Keys triggering too easily or not at all | Press `R` to recalibrate at your actual playing height |
| Hand not detected | Improve lighting; make sure your hands are fully in frame |
