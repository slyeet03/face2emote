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

    # Draw facial landmarks
    if results.face_landmarks:
        drawing_utils.draw_landmarks(
            image,
            results.face_landmarks,
            holistic.FACEMESH_CONTOURS,
            drawing_utils.DrawingSpec(
                color=(255, 0, 255), thickness=1, circle_radius=1
            ),
            drawing_utils.DrawingSpec(
                color=(0, 255, 255), thickness=1, circle_radius=1
            ),
        )

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

    return image
