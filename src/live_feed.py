#!/usr/bin/env python3
"""
Raspberry Pi Camera Module 3 Live Feed
Optimized for small grain detail (black pepper, papaya seeds, etc.)
"""

from picamera2 import Picamera2, Preview
from libcamera import controls
import time
import sys

# --- Macro Settings ---
MACRO_CONTROLS = {
    "AfMode":       controls.AfModeEnum.Continuous,  # Continuous autofocus
    "AfRange":      controls.AfRangeEnum.Macro,       # Macro focus range (~8-15cm)
    "AfSpeed":      controls.AfSpeedEnum.Fast,         # Fast focus response
    "Sharpness":    2.5,    # Enhance fine surface texture (default 1.0, max 16.0)
    "Contrast":     1.3,    # Improve edge definition between grain features
    "Saturation":   1.4,    # Richer color differentiation (black vs brown tones)
    "Brightness":   0.05,   # Slight boost to reveal shadow detail
    "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
}

def start_live_feed():
    """Start macro-optimized live camera feed"""
    print("Starting Camera Module 3 - Macro Live Feed")
    print("Optimized for: black pepper, papaya seeds, small grains\n")

    try:
        picam2 = Picamera2()

        # Preview config: good resolution for live macro inspection
        preview_config = picam2.create_preview_configuration(
            main={"format": "XRGB8888", "size": (1920, 1080)},
            controls=MACRO_CONTROLS
        )
        picam2.configure(preview_config)

        # Start Qt preview window
        print("Opening preview window...")
        print("Press Ctrl+C to stop\n")
        picam2.start_preview(Preview.QTGL)

        picam2.start()

        # Allow AF and exposure to settle
        time.sleep(2)

        print("✓ Live feed running - macro mode active")
        print(f"  Sharpness: {MACRO_CONTROLS['Sharpness']}")
        print(f"  Contrast:  {MACRO_CONTROLS['Contrast']}")
        print(f"  Focus:     Continuous Macro\n")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopping live feed...")
        picam2.stop_preview()
        picam2.stop()
        print("✓ Stopped.")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure display is connected (HDMI/DSI)")
        print("2. sudo apt install -y python3-pyqt5 python3-opengl")
        print("3. Check camera connection and raspi-config settings")
        return False

    return True

if __name__ == "__main__":
    success = start_live_feed()
    sys.exit(0 if success else 1)
