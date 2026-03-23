# Indoor Asthma Trigger — Audio Monitor

Continuously listens via microphone, detects cough / sneeze / wheeze events using YAMNet (Google, TensorFlow Hub), and writes episodes to Firebase Realtime Database.

---

## Requirements

- Python 3.10+
- A microphone connected to your machine
- A Firebase project with Realtime Database enabled

---

## Installation

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Firebase Setup — Getting the Service Account Key

1. Go to the [Firebase Console](https://console.firebase.google.com/) and open your project
2. Click the gear icon → **Project Settings**
3. Go to the **Service accounts** tab
4. Click **Generate new private key** → **Generate Key**
5. Save the downloaded JSON file as `serviceAccountKey.json` in the project directory

> **Keep this file private** — it grants full admin access to your Firebase project. Never commit it to git.

### Enable Realtime Database

1. In the Firebase Console, go to **Build → Realtime Database**
2. Click **Create database** → choose a region → start in **test mode** (you can lock down rules later)
3. Copy the database URL — it looks like `https://<your-project-id>-default-rtdb.firebaseio.com/`

---

## Configuration

Copy the template and fill in your values:

```bash
cp config.template.json config.json
```

Edit `config.json`:

```json
{
  "firebase_credentials_path": "serviceAccountKey.json",
  "firebase_database_url": "https://<YOUR-PROJECT-ID>-default-rtdb.firebaseio.com/",
  "firebase_collection": "episodes"
}
```

All other settings (thresholds, voting, microphone) have sensible defaults. See `config.template.json` for the full list with explanations.

---

## Usage

```bash
# Run with default microphone and default config
python monitor.py

# Run with a custom config file
python monitor.py --config config.json

# Show all available microphone devices
python monitor.py --list-devices

# Debug mode — prints top-5 YAMNet scores on every audio clip
# Use this to see what the model is hearing and tune thresholds
python monitor.py --debug-audio
```

---

## How It Works

```
Microphone
    │  (50ms frames)
    ▼
VAD (Voice Activity Detection)
    │  triggers when RMS energy > vad_energy_threshold
    │  assembles a 1s clip when activity ends
    ▼
YAMNet Classifier
    │  scores 521 audio classes
    │  filters to target classes: cough, sneeze, wheeze
    ▼
VotingBuffer
    │  requires min_votes hits within vote_window_s
    │  reduces false positives from brief noise
    ▼
EpisodeTracker
    │  groups detections within episode_gap_s into one episode
    │  closes episode after silence exceeds the gap
    ▼
Firebase Realtime Database
    └─ /episodes/{key} → episode record
```

### Episode record written to Firebase

```json
{
  "label":               "cough",
  "episode_start_utc":   "2025-09-01T14:01:00.123+00:00",
  "episode_end_utc":     "2025-09-01T14:01:08.456+00:00",
  "episode_start_epoch": 1725198060.123,
  "episode_end_epoch":   1725198068.456,
  "duration_s":          8.333,
  "detection_count":     6,
  "peak_confidence":     0.84,
  "source":              "yamnet"
}
```

---

## Tuning Guide

| Symptom | Fix |
|---|---|
| Nothing detected — mic too quiet | Lower `vad_energy_threshold` (default `0.005`) |
| Too many false positives | Raise `confidence_threshold` or `min_votes` |
| Real events missed | Lower `confidence_threshold` (default `0.15`) or set `min_votes: 1` |
| Episodes split into many short records | Raise `episode_gap_s` (default `4.0`) |
| High latency before detection | Lower `vad_hold_frames` (default `4`) |
| Want to see what YAMNet is scoring | Run with `--debug-audio` |

---

## Logs

The daemon logs at two levels:

- **INFO** — detections, episode start/end, Firebase writes
- **DEBUG** — heartbeat mic levels, VAD triggers, classifier scores, vote counts

To reduce noise, change the logging level in `monitor.py` line ~74:

```python
level=logging.INFO,   # only show detections and episodes
```

---

## File Structure

```
monitor.py               # main daemon
predictor.py             # ML predictor (separate process)
config.json              # your local config (not committed)
config.template.json     # config reference with all fields documented
predictor_config.json    # predictor config
serviceAccountKey.json   # Firebase credentials (not committed — keep private)
requirements.txt
```
