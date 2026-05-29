# Air Piano

Play a piano with your bare hands in the air — no keyboard required. Uses your webcam, MediaPipe hand tracking, and OpenCV. Tracks both hands independently with a two-tier height model that physically distinguishes white keys from black keys.

---

## Setup

**Requirements:** Python 3.10+, a webcam.

```bash
cd newproject
pip install -r requirements.txt
python air_piano.py
```

On first run the app downloads a ~2 MB hand-landmarker model automatically. If a saved `profile.json` exists from a previous session it loads instantly and skips straight to playing. Press `R` to retrain.

---

## The Two-Tier Height Model

On a real piano, black keys are physically elevated and set further back. Your fingers reach slightly higher and further inward to hit them. This app mirrors that:

| Tier | What the camera sees | Baseline line |
|------|---------------------|---------------|
| White keys | Finger at natural resting height | **Cyan line** |
| Black keys | Finger raised slightly, as if reaching for an elevated key | **Purple line** |

Both tiers are learned from you during training. The midpoint between them is the **tier boundary** — fingers above it trigger black keys, fingers below it trigger white keys (when the x-position falls in a black key zone).

Each fingertip is labelled **W** (white tier) or **B** (black tier) in real time so you can see which tier the system is reading.

---

## Phases (fully automatic — no keyboard required)

### Setup
Step back until your face and both hands are clearly visible. Once both hands are detected for 1.5 continuous seconds the app advances automatically.

### Calibration (4 seconds)
Hold both hands at the height you'd naturally hover above a piano. The app samples every fingertip's y-pixel position for 4 seconds and takes the 70th percentile as your **white key baseline** — the screen height of your resting hands.

### Training Round 1 — White Keys
All 8 white keys in random order. For each:
- The target key is highlighted in cyan on the piano
- Press it naturally at your normal resting height
- **5 presses required** — the key will not advance until you press it enough times
- If 10 seconds pass with zero presses the key is re-queued at the end of the round

The app records press depth for each (hand, key) combination.

### Training Round 2 — Black Keys
All 5 black keys in random order. For each:
- **Raise your finger slightly** as if reaching for a physically elevated key further back on a piano
- **7 presses required** — more data is needed because the motion is less practiced and the zones are narrower
- Same re-queue logic applies

The app records both the elevated resting y-position and the press depth.

### Playing
Play freely. All 10 fingers work simultaneously:
- **x-position** selects the key column
- **y-height** determines white vs black tier (cyan = white, purple = black)
- **pressing down** past your learned threshold fires the note
- Stats overlay in the top-right corner shows active notes and recent history

---

## Profile Persistence

After training completes, your profile is saved to `profile.json`. It stores:

| Field | What it is |
|-------|-----------|
| `white_baseline_y` | Pixel row of your resting hand height |
| `black_baseline_y` | Median pixel row of all black key approach heights |
| `tier_boundary` | Midpoint between the two baselines (the tier split) |
| `press_thresholds` | Per (hand, key) trigger depth in pixels |
| `black_resting_y` | Per (hand, key) elevated resting height for black keys |
| `frame_height` | Used to invalidate the profile if camera resolution changes |
| `trained_at` | Timestamp |

On next launch the profile is auto-loaded and play starts in 2.5 seconds. Press `R` to retrain.

---

## Keys

One octave: **C4 D4 E4 F4 G4 A4 B4 C5** (white) + **C#4 D#4 F#4 G#4 A#4** (black).

---

## Controls

| Key | Action |
|-----|--------|
| `R` | Restart from the beginning (clears session, keeps profile.json until retrain completes) |
| `Q` | Quit |

No other keyboard input needed. All phases advance automatically.

---

## Visual Reference

| Element | Meaning |
|---------|---------|
| Cyan horizontal line | White key baseline (natural resting height) |
| Purple horizontal line | Black key baseline (elevated tier height) |
| **W** label on fingertip | Finger detected at white key tier |
| **B** label on fingertip | Finger detected at black key tier |
| Rainbow column bands | Key zones — one colour per white key |
| Depth bar | Glowing vertical line from baseline to fingertip — brightens as you approach threshold |
| Ripple | Fires at the baseline each time a note triggers |
| Green key | Actively pressed |
| Cyan key | Target key during training |

---

## Tips

- **Good lighting** on your hands significantly improves tracking.
- During **black key training**, consciously raise your wrist as if reaching for a physically elevated key — the app is learning that height as the black key signal.
- The **W/B label** on each fingertip is your real-time feedback during play — if it shows the wrong tier, adjust your hand height.
- Press `R` to retrain if you move significantly closer or further from the camera, or if the profile feels off.
- Delete `profile.json` to force a full retrain from scratch.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No sound | `pip install sounddevice` |
| Webcam not opening | Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `air_piano.py` |
| Setup never advances | Improve lighting; step back so both hands and face fit in frame |
| Black and white keys confused | Retrain — raise your finger more distinctly during black key training |
| Training key never advances | You must reach the required press count; the timer alone won't skip a key |
| Notes firing on wrong tier | Watch the W/B label on your fingertip and adjust hand height |
| Profile loads but feels wrong | Delete `profile.json` and retrain |
