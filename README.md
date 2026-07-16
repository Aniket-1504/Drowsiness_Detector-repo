# AI Driver Monitoring System

Real-time driver safety monitoring system that detects drowsiness, yawning, and bad posture using MediaPipe and OpenCV. Triggers audio alerts and logs all incidents.

## Features

- Drowsiness detection (eyes closed 2s+)
- Yawn detection (mouth open 1.5s+)
- Bad posture monitoring (head tilt, head turn, slouching)
- Real-time audio alerts
- CSV logging of all incidents

## Tech Stack

MediaPipe FaceLandmarker | OpenCV | Python | pygame

## Installation

```bash
pip install mediapipe opencv-python pygame numpy
python drowsiness_detector_with_posture.py
```

## Usage

1. Place `alarm.mp3` in project folder
2. Run the script
3. Press 'q' to quit

## Output

`alert_log.csv` — Contains timestamp, alert type, and details for each incident

## How It Works

- MediaPipe detects facial landmarks and body pose
- Eye closure & yawn detection triggers alerts
- Posture analyzer checks head position & shoulder alignment
- Audio alarm plays on detection with cooldown

## Author

**Aniket Kadam**  
SITS, Pune (SPPU)

---

**Status:** ✅ Completed (July 2026)
