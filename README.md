
# Air Piano

Play piano in the air with your webcam — no keyboard required.

## Setup

```bash
pip install -r requirements.txt
python air_piano.py
```

## How it works

| Phase | What happens |
|-------|-------------|
| **Calibration** (4 s) | Hold your hands at your natural playing height. The app records where your fingertips rest. |
| **Playing** | A cyan baseline appears at your rest height. Press any finger **down past it** and the note plays. |

The piano spans the full camera width — your **x-position** picks the key, your **downward motion** triggers it.

## Controls

| Key | Action |
|-----|--------|
| `R` | Re-run calibration |
| `Q` | Quit |

## Layout

One octave: **C4 → C5** (8 white keys + 5 sharps/flats).

Rainbow-coloured hover zones show each note's column. The mini keyboard in the right panel lights up as you play.

## Requirements

- Python 3.10+
- Webcam
- `opencv-python`, `mediapipe`, `numpy`, `sounddevice`
