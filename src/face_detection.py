import cv2 as cv
import mediapipe as mp

holistic = mp.solutions.holistic
drawing_utils = mp.solutions.drawing_utils

holistic_model = holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)


def face_detect(image):
    image.flags.writeable = False
    results = holistic_model.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    image.flags.writeable = True

    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    face_roi = image
    x_min, x_max = int(0), int(0)
    y_min, y_max = int(0), int(0)
    landmarks = 0
    # Draw facial landmarks
    if results.face_landmarks:
        h, w, _ = image.shape
        x_coords = [lm.x * w for lm in results.face_landmarks.landmark]
        y_coords = [lm.y * h for lm in results.face_landmarks.landmark]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        # Crop face ROI
        face_roi = image[y_min:y_max, x_min:x_max]

    # Draw right hand landmarks
    if results.right_hand_landmarks:
        drawing_utils.draw_landmarks(
            image, results.right_hand_landmarks, holistic.HAND_CONNECTIONS
        )

    # Draw left hand landmarks
    if results.left_hand_landmarks:
        drawing_utils.draw_landmarks(
            image, results.left_hand_landmarks, holistic.HAND_CONNECTIONS
        )

    return face_roi, (x_min, y_min, x_max, y_max), image, landmarks

