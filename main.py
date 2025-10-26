import cv2 as cv

from src.camera import capture_frames
from src.face_detection import face_detect

for frame in capture_frames():
    image = face_detect(frame)
    cv.imshow("marks", image)

    if cv.waitKey(5) & 0xFF == ord("q"):
        break
    pass
