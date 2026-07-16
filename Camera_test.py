"""
Minimal webcam test - isolates whether OpenCV can read frames at all,
separate from MediaPipe/alerts. Run this first to debug camera issues.

Always forces the DirectShow (CAP_DSHOW) backend on Windows, and tries
multiple camera indexes since laptops often have more than one camera
device (e.g., an IR camera for Windows Hello alongside the regular webcam).
"""

import cv2

cap = None
working_index = None

for index in [0, 1, 2, 3]:
    print(f"Trying camera index {index} with CAP_DSHOW...")
    test_cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not test_cap.isOpened():
        print(f"  Index {index}: could not open.")
        test_cap.release()
        continue

    success, frame = test_cap.read()
    if success and frame is not None and frame.mean() > 1.0:
        print(f"  Index {index}: WORKS (brightness={frame.mean():.2f})")
        cap = test_cap
        working_index = index
        break
    else:
        brightness = frame.mean() if success and frame is not None else "N/A"
        print(f"  Index {index}: opened but frame is black/empty "
              f"(brightness={brightness}).")
        test_cap.release()

if cap is None:
    print("\nNo working camera found on indexes 0-3 with CAP_DSHOW.")
    print("Check: physical camera cover/shutter, Windows camera privacy "
          "settings (Settings > Privacy & Security > Camera > allow desktop "
          "apps), or another app currently holding the camera.")
else:
    print(f"\nUsing camera index {working_index}. Press 'q' in the window to quit.")

    frame_count = 0
    fail_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            fail_count += 1
            print(f"Frame read failed ({fail_count} times so far)")
            if fail_count > 30:
                print("Too many consecutive failures - stopping.")
                break
            continue

        frame_count += 1
        fail_count = 0

        cv2.putText(frame, f"Frame #{frame_count} (index {working_index})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Camera Test - press q to quit", frame)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()

cv2.destroyAllWindows()