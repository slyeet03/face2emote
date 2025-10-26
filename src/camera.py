import cv2 as cv
import numpy as np


def capture_frames():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't recieve frame")
                break

            processed_frame = process_frames(frame)
            yield processed_frame
    finally:
        cap.release()
        cv.destroyAllWindows()


def process_frames(frame):
    frame=cv.resize(frame,(800,600))
    image = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    return image
