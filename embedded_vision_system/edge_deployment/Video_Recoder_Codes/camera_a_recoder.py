#!/usr/bin/python3

import time
import threading
import argparse
import cv2
import os
import RPi.GPIO as gp
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput


def select_camera_a():
    """Configure the multiplexer to enable only Camera A."""
    gp.setwarnings(False)
    gp.setmode(gp.BOARD)
    gp.setup(7, gp.OUT)
    gp.setup(11, gp.OUT)
    gp.setup(12, gp.OUT)

    # Set I2C to select channel 0x04 (Camera A)
    os.system("i2cset -y 1 0x70 0x00 0x04")

    # Set GPIO pins to activate Camera A
    gp.output(7, False)
    gp.output(11, False)
    gp.output(12, True)


def show_preview(picam2, stop_event):
    while not stop_event.is_set():
        frame = picam2.capture_array("main")  # Already in BGR888 format
        cv2.imshow("Recording - Press 'q' to Stop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Record video using IMX500.")
    parser.add_argument('--output', type=str, default='video.mp4', help='Output filename')
    parser.add_argument('--bitrate', type=int, default=4000000, help='Encoding bitrate (default 4 Mbps)')
    parser.add_argument('--width', type=int, default=1280, help='Video width (default 1280)')
    parser.add_argument('--height', type=int, default=720, help='Video height (default 720)')
    parser.add_argument('--fps', type=int, default=24, help='Frames per second (default 24)')
    parser.add_argument('--audio', action='store_true', help='Enable audio (requires microphone setup)')
    parser.add_argument('--headless', action='store_true', help='Run without preview window; stop with Ctrl+C')
    args = parser.parse_args()

    try:
        # Select Camera A via multiplexer
        select_camera_a()

        # Initialize camera
        picam2 = Picamera2()

        video_config = picam2.create_video_configuration(
            main={"size": (args.width, args.height), "format": "BGR888"},
            controls={
                "FrameRate": args.fps,
                "FrameDurationLimits": (int(1e6 / args.fps), int(1e6 / args.fps))
            }
        )
        picam2.configure(video_config)

        encoder = H264Encoder(bitrate=args.bitrate)
        output = FfmpegOutput(args.output, audio=args.audio)

        if args.headless:
            print("Headless recording started. Press Ctrl+C to stop.")
            picam2.start_recording(encoder, output)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping recording...")
                picam2.stop_recording()
                print(f"✅ Recording saved to: {args.output}")
        else:
            stop_event = threading.Event()
            preview_thread = threading.Thread(target=show_preview, args=(picam2, stop_event))
            preview_thread.start()

            print("Recording started. Press 'q' in the preview window to stop.")
            picam2.start_recording(encoder, output)

            while not stop_event.is_set():
                time.sleep(0.1)

            picam2.stop_recording()
            print(f"✅ Recording saved to: {args.output}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()