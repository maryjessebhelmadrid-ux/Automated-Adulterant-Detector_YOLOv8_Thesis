#!/usr/bin/env python3
"""
Raspberry Pi Camera Module 3 Test Script
Tests camera functionality on Raspberry Pi 5
"""

from picamera2 import Picamera2
import time
import sys

def test_camera():
    """Test the Raspberry Pi Camera Module 3"""
    print("Initializing Raspberry Pi Camera Module 3...")

    try:
        # Create camera instance
        picam2 = Picamera2()

        # Get camera configuration
        print("\nCamera detected successfully!")
        camera_config = picam2.create_still_configuration()
        picam2.configure(camera_config)

        # Print camera properties
        print("\nCamera Properties:")
        print(f"  Sensor: {picam2.camera_properties.get('Model', 'Unknown')}")
        print(f"  Resolution: {picam2.camera_properties.get('PixelArraySize', 'Unknown')}")

        # Start camera
        print("\nStarting camera...")
        picam2.start()

        # Let camera warm up
        print("Warming up camera (2 seconds)...")
        time.sleep(2)

        # Capture test image
        output_file = "test_image.jpg"
        print(f"\nCapturing test image to: {output_file}")
        picam2.capture_file(output_file)

        print(f"✓ Image captured successfully: {output_file}")

        # Stop camera
        picam2.stop()
        print("\nCamera test completed successfully!")
        print("You can view the captured image: test_image.jpg")

        return True

    except Exception as e:
        print(f"\n✗ Camera test failed!")
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure camera is properly connected to the camera port")
        print("2. Check if camera is enabled: sudo raspi-config")
        print("3. Install required library: sudo apt install -y python3-picamera2")
        print("4. Reboot after enabling camera: sudo reboot")
        return False

if __name__ == "__main__":
    success = test_camera()
    sys.exit(0 if success else 1)
