import os

EMOTE_DIR = "assets/emotes"


def get_emote_path(emotion=None, gesture=None):
    if gesture:
        gesture_path = os.path.join(EMOTE_DIR, f"{gesture}.png")
        if os.path.exists(gesture_path):
            return gesture_path

    if emotion:
        emotion_path = os.path.join(EMOTE_DIR, f"{emotion.lower()}.png")
        if os.path.exists(emotion_path):
            return emotion_path

    return os.path.join(EMOTE_DIR, "none.png")
