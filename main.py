import cv2 as cv

from src.camera import capture_frames
from src.display import show_display
from src.emote_mapper import get_emote_path
from src.emotion_recognition import predict
from src.face_detection import face_detect
from src.gesture_recognition import get_hand_gesture
from utils.config import LABELS
from utils.sound import play_sound

prev_emotion = None
for frame in capture_frames():
    face_roi, bbox, image, left_hand, right_hand = face_detect(frame)
    gesture = get_hand_gesture(left_hand, right_hand)

    emotion = None  # default

    if gesture:
        emote_path = get_emote_path(None, gesture)

    elif face_roi is not None and bbox is not None:
        label_idx, confidence = predict(face_roi)
        emotion = LABELS[label_idx]

        x_min, y_min, x_max, y_max = bbox
        cv.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv.putText(
            image,
            emotion,
            (x_min, y_max + 25),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )

        emote_path = get_emote_path(emotion, gesture)
    else:
        emote_path = get_emote_path(None, None)

    show_display(image, emote_path)

    if emotion and emotion != prev_emotion:
        play_sound(emotion)
        prev_emotion = emotion

    if cv.waitKey(5) & 0xFF == ord("q"):
        break

cv.destroyAllWindows()
