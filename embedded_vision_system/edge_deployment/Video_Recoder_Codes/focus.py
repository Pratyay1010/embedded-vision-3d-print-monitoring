#!/usr/bin/python3

import cv2
import os
import RPi.GPIO as gp
from picamera2 import Picamera2

def select_camera_a():
    """Configure the multiplexer to enable only Camera A."""
    gp.setwarnings(False)
    gp.setmode(gp.BOARD)
    gp.setup(7, gp.OUT)
    gp.setup(11, gp.OUT)
    gp.setup(12, gp.OUT)

    os.system("i2cset -y 1 0x70 0x00 0x04")  # Select channel 0x04 for Camera A

    gp.output(7, False)
    gp.output(11, False)
    gp.output(12, True)

def main():
    # Select Camera A
    select_camera_a()

    # Initialize and configure the camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1280, 720), "format": "BGR888"})
    picam2.configure(config)
    picam2.start()

    print("Live preview started (Camera A). Press 'q' to quit.")

    try:
        while True:
            frame = picam2.capture_array()
            cv2.imshow("Live Preview - Focus Test (Camera A)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        picam2.stop()
        print("Preview stopped.")

if __name__ == "__main__":
    main()