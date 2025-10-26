import cv2 as cv

from src.camera import capture_frames
from src.emotion_recognition import predict
from src.face_detection import face_detect
from utils.config import LABELS

for frame in capture_frames():
    face_roi, bbox, image, landmark = face_detect(frame)

    if face_roi is not None and bbox is not None:
        label_idx, confidence = predict(face_roi)
        emotion = LABELS[label_idx]

        # unpack coordinates
        x_min, y_min, x_max, y_max = bbox

        # draw red rectangle
        cv.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # draw emotion text below the box
        text_y = y_max + 25 if y_max + 25 < image.shape[0] else y_max - 10
        cv.putText(
            image,
            f"{emotion} ({confidence*100:.1f}%)",
            (x_min, text_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )

        cv.imshow("Emotion Detection", image)

    if cv.waitKey(5) & 0xFF == ord("q"):
        break

cv.destroyAllWindows()
