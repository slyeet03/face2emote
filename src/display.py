import cv2 as cv
import numpy as np


def show_display(frame, emote_path, window_name="Face2Emote"):
    display_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  # add this
    
    emote_img = cv.imread(emote_path)
    if emote_img is None:
        emote_img = np.zeros_like(display_frame)

    emote_img = cv.resize(emote_img, (display_frame.shape[1], display_frame.shape[0]))
    combined = np.hstack((display_frame, emote_img))

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 1280, 480)
    cv.imshow(window_name, combined)