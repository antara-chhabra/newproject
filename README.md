# Air Piano

Play a piano with your bare hands in the air — no keyboard required. Uses your webcam, MediaPipe hand tracking, and OpenCV. Supports two hands with personalized press calibration.

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

The app runs through 4 phases:

### Phase 1 — Setup (interactive)
Move away from the screen until your **entire face AND both hands** are clearly visible in the frame. The app shows you a hand counter; press SPACE when ready.

### Phase 2 — Calibration (4 seconds)
Hold both hands at the height you'd naturally hover above a piano — like you're about to play. The app records your resting finger height as your **baseline**. Stay still.

### Phase 3 — Training (guided, ~40 seconds)
The app will ask you to press each key one by one. Press each key multiple times while the app learns **how you press**. For each key and hand combination, the app measures your typical press depth and sets a personalized threshold.

On the screen:
- The current key is highlighted in the piano at the bottom
- Your fingers are shown with colour-coded dots
- A vertical **depth bar** glows from baseline to fingertip
- Use this bar to calibrate your feel for the depth

Press SPACE to move to the next key.

### Phase 4 — Playing
Play freely! All 10 fingers work simultaneously. Your **left-right position** picks the key, your **downward motion** triggers it. The dashboard on the right shows active notes and a fading history.

---

## Keys

One octave: **C4 D4 E4 F4 G4 A4 B4 C5** (white keys) + **C#4 D#4 F#4 G#4 A#4** (black keys, narrower zones).

---

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Advance to next phase / next training key |
| `R` | Reset to setup (re-run all phases) |
| `Q` | Quit |

---

## How It Works

**Two-hand mapping:** The app tracks both your left and right hand separately. Each hand learns its own pressing pattern for each key — so if you press C4 with your left thumb vs right index, the app adapts to each.

**Personalized thresholds:** During training, the app records how deep you press each key. It uses the 40th percentile of your press depths as your personalized threshold — so you don't have to press as hard if you naturally press gently, or can press harder if that's your style.

**Real-time visual feedback:** The depth bar shows you exactly how close you are to triggering a note. Watch the bar grow and brighten as you press deeper.

---

## Tips

- **Good lighting** on your hands makes tracking much more reliable.
- **Don't shy away during training** — press each key firmly and naturally. The more varied your presses, the better the personalization.
- In playing mode, the thresholds are **fixed** to what you trained. Press `R` to recalibrate if your position changes drastically.
- Use the **depth bar** to get a tactile sense of the press depth before committing to playing scales.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No sound | `pip install sounddevice` |
| Webcam not opening | Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `air_piano.py` |
| Hands not detected during setup | Improve lighting; make sure both hands are fully in frame |
| Training keys feel too sensitive / not sensitive enough | Re-run training (`R`) and press more firmly or gently as needed |
| One hand not being tracked | Face the camera squarely; keep both hands fully visible and in good light |
