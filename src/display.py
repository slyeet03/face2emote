import cv2 as cv
import numpy as np


def show_display(frame, emote_path, window_name="Face2Emote"):
    emote_img = cv.imread(emote_path)
    if emote_img is None:
        emote_img = np.zeros_like(frame)

    emote_img = cv.resize(emote_img, (frame.shape[1], frame.shape[0]))

    combine = np.hstack((frame, emote_img))

    cv.imshow(window_name, combine)
    cv.namedWindow("Face2Emote", cv.WINDOW_NORMAL)
    cv.resizeWindow("Face2Emote", 1280, 480)
