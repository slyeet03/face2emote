import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions, RunningMode
import numpy as np

options = HolisticLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="assets/holistic_landmarker.task"),
    running_mode=RunningMode.IMAGE,
    min_face_detection_confidence=0.5,
)

holistic_model = HolisticLandmarker.create_from_options(options)


def face_detect(image):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    results = holistic_model.detect(mp_image)

    face_roi = image
    x_min, x_max, y_min, y_max = 0, 0, 0, 0

    if results.face_landmarks:
        h, w, _ = image.shape
        x_coords = [lm.x * w for lm in results.face_landmarks]
        y_coords = [lm.y * h for lm in results.face_landmarks]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        face_roi = image[y_min:y_max, x_min:x_max]

    left_hand = results.left_hand_landmarks if results.left_hand_landmarks else None
    right_hand = results.right_hand_landmarks if results.right_hand_landmarks else None

    return (
        face_roi,
        (x_min, y_min, x_max, y_max),
        image,
        left_hand,
        right_hand,
    )