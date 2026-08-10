# AI Driver Monitoring System
Real-time driver safety monitoring system that detects drowsiness, yawning, and bad posture using MediaPipe and OpenCV. Triggers audio alerts and logs all incidents.
## Features
- Drowsiness detection (eyes closed 0.1s+)
- Yawn detection (mouth open 1.5s+)
- Bad posture monitoring (head tilt, head turn, slouching)
- Real-time audio alerts with automatic beep-tone fallback if no alarm file is found
- CSV logging of all incidents
## Tech Stack
MediaPipe FaceLandmarker | OpenCV | Python | pygame | NumPy
## Installation
```bash
pip install mediapipe opencv-python pygame numpy
python drowsiness_detector_with_posture.py
```
## Usage
1. Run the script
2. Press 'q' to quit
## Output
`alert_log.csv` — Contains timestamp, alert type, and details for each incident
## How It Works
- MediaPipe detects facial landmarks and body pose
- Eye closure & yawn detection triggers alerts
- Posture analyzer checks head position & shoulder alignment
- Audio alarm plays on detection with cooldown — uses `alarm.mp3` if present, otherwise falls back to a generated sine-wave beep
## Author
**Aniket Kadam**  
SITS, Pune (SPPU)
---
**Status:** ✅ Completed (July 2026)
