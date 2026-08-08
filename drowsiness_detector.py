"""
AI Driver Monitoring System - Week 1
Live webcam + MediaPipe FaceLandmarker (Tasks API) + blendshape-based
drowsiness (eye closure) and yawning detection.

NOTE: MediaPipe's Python package recently dropped the old `mp.solutions.face_mesh`
API in favor of the newer Tasks API used here. This script targets mediapipe>=0.10.

Run:
    pip install mediapipe opencv-python numpy
    python drowsiness_detector.py

First run will auto-download the face_landmarker.task model (~a few MB).
Press 'q' to quit.
"""

import cv2
import time
import os
import csv
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import pygame

# ---------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------
MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

EYE_BLINK_THRESHOLD = 0.5     # blendshape score above this = eye considered closed
EYE_CLOSED_SECONDS = 0.1      # how long eyes must stay closed to trigger alert

MOUTH_OPEN_THRESHOLD = 0.5    # blendshape score above this = mouth wide open
YAWN_SECONDS = 1.5            # how long mouth must stay open to count as a yawn

# --- Week 2 additions ---
ALARM_FILE = "alarm.mp3"      # put your alarm sound file here, same folder as script
ALERT_COOLDOWN_SECONDS = 4.0  # min gap between repeated alarm sounds
LOG_FILE = "alert_log.csv"    # every triggered alert gets a row here

# --- Beep fallback (used automatically if ALARM_FILE is missing) ---
BEEP_FREQ_HZ = 1000            # tone pitch
BEEP_DURATION_SECONDS = 0.6    # tone length


def make_beep_sound(freq_hz=BEEP_FREQ_HZ, duration_s=BEEP_DURATION_SECONDS,
                     sample_rate=44100, volume=0.5):
    """Generate a simple sine-wave beep as a pygame Sound object (no file needed)."""
    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    wave = np.sin(2 * np.pi * freq_hz * t)

    # short fade in/out to avoid clicks at start/end
    fade_len = int(sample_rate * 0.02)
    if fade_len > 0:
        fade = np.linspace(0, 1, fade_len)
        wave[:fade_len] *= fade
        wave[-fade_len:] *= fade[::-1]

    wave = (wave * volume * 32767).astype(np.int16)
    stereo_wave = np.column_stack([wave, wave])  # duplicate to 2 channels
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo_wave))


# ---------------------------------------------------------
# 2. MODEL DOWNLOAD (first run only)
# ---------------------------------------------------------
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face_landmarker model (first run only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


# ---------------------------------------------------------
# 3. HELPER: pull a named blendshape score out of the result
# ---------------------------------------------------------
def get_blendshape_score(face_blendshapes, name):
    for category in face_blendshapes:
        if category.category_name == name:
            return category.score
    return 0.0


# ---------------------------------------------------------
# 4. ALERT MANAGER (sound + cooldown + logging)
# ---------------------------------------------------------
class AlertManager:
    def __init__(self, sound_path, cooldown_seconds, log_path):
        self.cooldown_seconds = cooldown_seconds
        self.log_path = log_path
        self.last_alert_time = {}  # per alert type, e.g. "drowsiness", "yawn"

        self.sound_loaded = False
        self.use_file = False
        self.beep_sound = None
        try:
            pygame.mixer.init()
            if os.path.exists(sound_path):
                pygame.mixer.music.load(sound_path)
                pygame.mixer.music.set_volume(1.0)
                self.sound_loaded = True
                self.use_file = True
                print(f"Alarm sound loaded successfully: {sound_path}")
            else:
                # Fall back to a generated beep tone so alerts still have sound
                self.beep_sound = make_beep_sound()
                self.sound_loaded = True
                self.use_file = False
                print(f"WARNING: '{sound_path}' not found. "
                      f"Using a generated beep tone instead.")
        except Exception as e:
            print(f"WARNING: Audio unavailable ({e}). Alerts will still be "
                  f"shown on screen and logged, just without sound.")

        # Create log file with header if it doesn't exist yet
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "alert_type"])

    def trigger(self, alert_type):
        """Play sound + log, but only if cooldown has passed for this alert type."""
        now = time.time()
        last = self.last_alert_time.get(alert_type, 0)

        if now - last >= self.cooldown_seconds:
            self.last_alert_time[alert_type] = now

            if self.sound_loaded:
                try:
                    if self.use_file:
                        pygame.mixer.music.play()
                    else:
                        self.beep_sound.play()
                    print(f"[AUDIO] Playing alarm for: {alert_type}")
                except Exception as e:
                    print(f"[AUDIO ERROR] Could not play sound: {e}")

            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), alert_type])

            print(f"[ALERT] {alert_type} logged at {time.strftime('%H:%M:%S')}")


# ---------------------------------------------------------
# 5. MAIN LOOP
# ---------------------------------------------------------
def main():
    ensure_model()
    alert_manager = AlertManager(ALARM_FILE, ALERT_COOLDOWN_SECONDS, LOG_FILE)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Try changing the index (0, 1, 2...).")
        return

    eyes_closed_start = None
    mouth_open_start = None
    drowsy_alert = False
    yawn_alert = False

    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)  # mirror view, feels natural
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        status_text = "No face detected"
        status_color = (0, 165, 255)  # orange

        if result.face_blendshapes:
            blendshapes = result.face_blendshapes[0]

            left_blink = get_blendshape_score(blendshapes, "eyeBlinkLeft")
            right_blink = get_blendshape_score(blendshapes, "eyeBlinkRight")
            avg_blink = (left_blink + right_blink) / 2.0

            jaw_open = get_blendshape_score(blendshapes, "jawOpen")

            # ---- Eye closure / drowsiness ----
            if avg_blink > EYE_BLINK_THRESHOLD:
                if eyes_closed_start is None:
                    eyes_closed_start = time.time()
                closed_duration = time.time() - eyes_closed_start

                if closed_duration >= EYE_CLOSED_SECONDS:
                    drowsy_alert = True
                    status_text = "DROWSINESS ALERT! Eyes closed"
                    status_color = (0, 0, 255)
                    alert_manager.trigger("drowsiness")
                else:
                    status_text = f"Eyes closing... ({closed_duration:.1f}s)"
                    status_color = (0, 255, 255)
            else:
                eyes_closed_start = None
                drowsy_alert = False
                status_text = "Eyes open - Alert"
                status_color = (0, 255, 0)

            # ---- Yawning ----
            if jaw_open > MOUTH_OPEN_THRESHOLD:
                if mouth_open_start is None:
                    mouth_open_start = time.time()
                open_duration = time.time() - mouth_open_start

                if open_duration >= YAWN_SECONDS:
                    yawn_alert = True
                    alert_manager.trigger("yawn")
            else:
                mouth_open_start = None
                yawn_alert = False

            if yawn_alert:
                cv2.putText(
                    frame, "YAWN DETECTED", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                )

            # Debug readout (useful for tuning thresholds)
            cv2.putText(
                frame, f"Blink: {avg_blink:.2f}  Jaw: {jaw_open:.2f}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

        # Status banner
        cv2.putText(
            frame, status_text, (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2,
        )

        if drowsy_alert or yawn_alert:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

        cv2.imshow("AI Driver Monitoring - Week 2: Alerts + Logging", frame)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()