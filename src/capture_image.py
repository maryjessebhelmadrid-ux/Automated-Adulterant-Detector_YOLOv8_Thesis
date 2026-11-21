#!/usr/bin/env python3
"""
Raspberry Pi Camera Module 3 Image Capture
Optimized for small grain detail (black pepper, papaya seeds, etc.)
SPACE to capture | Q to quit | No restart needed between captures
"""

from picamera2 import Picamera2
from libcamera import controls
import cv2
import time
import os
import sys
from datetime import datetime

# --- Settings ---
CAPTURE_INTERVAL_MIN = 0.5   # Minimum seconds between captures (debounce)

CAPTURE_RESOLUTION = (2304, 1296)   # High quality, stable frame rate
PREVIEW_RESOLUTION = (640,  360)    # Lores stream for smooth live display

# --- Macro Controls ---
MACRO_CONTROLS = {
    "AfMode":       controls.AfModeEnum.Continuous,
    "AfRange":      controls.AfRangeEnum.Macro,
    "AfSpeed":      controls.AfSpeedEnum.Fast,
    "Sharpness":    2.5,
    "Contrast":     1.3,
    "Saturation":   1.4,
    "Brightness":   0.05,
    "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
}

def capture_images():
    print("Camera Module 3 - Macro Capture Mode")
    print("Optimized for: black pepper, papaya seeds, small grains\n")

    try:
        picam2 = Picamera2()

        # Dual-stream config: lores for live display, main for saving
        # No mode switching needed — both streams run simultaneously
        config = picam2.create_video_configuration(
            main={"format": "RGB888", "size": CAPTURE_RESOLUTION},
            lores={"format": "YUV420", "size": PREVIEW_RESOLUTION},
            controls=MACRO_CONTROLS
        )
        picam2.configure(config)
        picam2.start()

        print("Stabilizing autofocus and exposure...")
        time.sleep(2)

        print(f"\n✓ Camera ready! Macro mode active.")
        print(f"  Capture resolution: {CAPTURE_RESOLUTION[0]}x{CAPTURE_RESOLUTION[1]}")
        print(f"  SPACE  - Capture image")
        print(f"  Q      - Quit\n")

        os.makedirs("captures", exist_ok=True)
        count = 0
        last_capture_time = 0

        while True:
            # Display from lores (fast, low CPU)
            lores_frame = picam2.capture_array("lores")
            preview = cv2.cvtColor(lores_frame, cv2.COLOR_YUV420p2BGR)

            # Overlay
            cv2.putText(preview, "SPACE: Capture | Q: Quit", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(preview, f"Captured: {count}", (10, PREVIEW_RESOLUTION[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            cv2.imshow("Camera - Macro Mode", preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print("\nQuitting...")
                break

            elif key == ord(' '):
                now = time.time()
                if now - last_capture_time < CAPTURE_INTERVAL_MIN:
                    continue  # Debounce rapid presses

                count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captures/macro_{timestamp}_{count:04d}.jpg"

                # Capture from main stream — no mode switch, camera keeps running
                main_frame = picam2.capture_array("main")
                main_bgr = cv2.cvtColor(main_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, main_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"✓ [{count:04d}] Saved: {filename}")

                last_capture_time = time.time()

                # Flash feedback
                flash = preview.copy()
                flash[:] = (255, 255, 255)
                cv2.imshow("Camera - Macro Mode", flash)
                cv2.waitKey(100)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. pip install opencv-python")
        print("2. Ensure camera is connected and enabled in raspi-config")
        return False

    finally:
        cv2.destroyAllWindows()
        try:
            picam2.stop()
        except Exception:
            pass
        print(f"\n✓ Done! {count} image(s) saved to captures/")

    return True

if __name__ == "__main__":
    success = capture_images()
    sys.exit(0 if success else 1)
