#!/usr/bin/python3

import time
import argparse
import numpy as np
import cv2
import os
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from datetime import datetime


def configureMultiCamera():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)

    GPIO.setup(7, GPIO.OUT)
    GPIO.setup(11, GPIO.OUT)
    GPIO.setup(12, GPIO.OUT)
    GPIO.setup(15, GPIO.OUT)
    GPIO.setup(16, GPIO.OUT)
    GPIO.setup(21, GPIO.OUT)
    GPIO.setup(22, GPIO.OUT)

    GPIO.output(11, True)
    GPIO.output(12, True)
    GPIO.output(15, True)
    GPIO.output(16, True)
    GPIO.output(21, True)
    GPIO.output(22, True)


def enableCamera(num):
    if num == 1:
        GPIO.output(7, False)
        GPIO.output(11, False)
        GPIO.output(12, True)
    elif num == 2:
        GPIO.output(7, True)
        GPIO.output(11, False)
        GPIO.output(12, True)
    elif num == 3:
        GPIO.output(7, False)
        GPIO.output(11, True)
        GPIO.output(12, False)
    elif num == 4:
        GPIO.output(7, True)
        GPIO.output(11, True)
        GPIO.output(12, False)


def disableMultiCamera():
    GPIO.output(7, False)
    GPIO.output(11, False)
    GPIO.output(12, True)


def displayImage(images, recording_flags):
    if not images:
        return

    height, width, _ = images[0].shape
    blank = np.zeros((height, width, 3), dtype=np.uint8)

    while len(images) < 4:
        images.append(blank)
        recording_flags.append(False)

    labels = ['Cam 1', 'Cam 2', 'Cam 3', 'Cam 4']
    for i in range(len(images)):
        label = labels[i]
        if recording_flags[i]:
            label += " [REC]"
        cv2.putText(images[i], label, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    top = np.hstack((images[0], images[1]))
    bottom = np.hstack((images[2], images[3]))
    combined = np.vstack((top, bottom))

    cv2.imshow("Multi-Camera View", combined)


def test_camera_channel(cam_num, width, height):
    try:
        enableCamera(cam_num)
        time.sleep(0.4)

        picam2 = Picamera2()
        config = picam2.create_still_configuration(
            main={"size": (width, height), "format": "BGR888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(0.4)

        frame = picam2.capture_array("main")
        picam2.stop()
        picam2.close()

        return frame is not None and frame.shape[0] > 0
    except Exception as e:
        print(f"[Cam {cam_num}] test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Multi-camera viewer/recorder for Raspberry Pi")
    parser.add_argument('--cams', type=int, default=0, help='Number of cameras to use (1–4). Auto-detect if 0.')
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')
    parser.add_argument('--fps', type=int, default=15, help='Recording FPS')
    parser.add_argument('--headless', action='store_true', help='Run without GUI')
    parser.add_argument('--record', action='store_true', help='Start recording immediately')
    args = parser.parse_args()

    os.makedirs("recordings", exist_ok=True)
    configureMultiCamera()

    if args.cams == 0:
        print("Detecting connected cameras...")
        connected_cams = []
        for cam in range(1, 5):
            if test_camera_channel(cam, args.width, args.height):
                connected_cams.append(cam)
        print(f"✅ Detected cameras: {connected_cams}")
    else:
        connected_cams = list(range(1, min(5, args.cams + 1)))
        print(f"Using specified cameras: {connected_cams}")

    recording = args.record
    video_writers = [None] * 4
    recording_flags = [False] * 4
    codec = cv2.VideoWriter_fourcc(*'XVID')

    picam2 = Picamera2()

    try:
        while True:
            start_time = time.time()
            frames = []

            for cam in connected_cams:
                enableCamera(cam)
                time.sleep(0.3)
                try:
                    picam2.stop()
                except:
                    pass

                try:
                    config = picam2.create_still_configuration(
                        main={"size": (args.width, args.height), "format": "BGR888"}
                    )
                    picam2.configure(config)
                    picam2.start()
                    time.sleep(0.3)
                    frame = picam2.capture_array("main")
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(f"[Cam {cam}] Capture error: {e}")
                    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)

                frames.append(frame)

                if recording:
                    idx = cam - 1
                    if video_writers[idx] is None:
                        filename = f"recordings/cam{cam}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                        video_writers[idx] = cv2.VideoWriter(filename, codec, args.fps, (args.width, args.height))
                        recording_flags[idx] = True
                        print(f"Started recording cam {cam} → {filename}")
                    video_writers[idx].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if not args.headless:
                displayImage(frames, recording_flags)

            key = cv2.waitKey(1) & 0xFF if not args.headless else -1
            if key == 27:  # ESC
                print("ESC pressed, exiting...")
                break
            elif key == ord('r'):
                recording = not recording
                print("Recording started" if recording else "Recording stopped")
                if not recording:
                    for vw in video_writers:
                        if vw:
                            vw.release()
                    video_writers = [None] * 4
                    recording_flags = [False] * 4
            elif key == ord('s'):
                for i, frame in enumerate(frames):
                    snap = f"recordings/cam{connected_cams[i]}_{datetime.now().strftime('%H%M%S')}.jpg"
                    cv2.imwrite(snap, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    print(f"Saved snapshot: {snap}")

            elapsed = time.time() - start_time
            print(f"Time: {elapsed:.2f}s | FPS: {1/elapsed:.2f}")

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Stopping recording...")

    finally:
        disableMultiCamera()
        GPIO.cleanup()
        try:
            picam2.stop()
            picam2.close()
        except:
            pass
        for vw in video_writers:
            if vw:
                vw.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print("Cleanup done. Goodbye!")


if __name__ == "__main__":
    main()
