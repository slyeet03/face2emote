import math


def get_hand_gesture(left_hand, right_hand):
    if left_hand and right_hand:
        left_gesture = detect_basic_gesture(left_hand)
        right_gesture = detect_basic_gesture(right_hand)

        if left_gesture and left_gesture == right_gesture:
            return left_gesture

    elif left_hand:
        return detect_basic_gesture(left_hand)
    elif right_hand:
        return detect_basic_gesture(right_hand)

    return None


def detect_basic_gesture(hand_landmarks):
    # 4 = Thumb tip, 8 = Index tip
    # 12 = Middle tip, 16 = Ring tip, 20 = Pinky tip
    # 2 = Thumb MCP (base joint)

    thumb_tip = hand_landmarks.landmark[4].y
    thumb_mcp = hand_landmarks.landmark[2].y
    index_tip = hand_landmarks.landmark[8].y
    middle_tip = hand_landmarks.landmark[12].y
    ring_tip = hand_landmarks.landmark[16].y
    pinky_tip = hand_landmarks.landmark[20].y

    # 👍 Thumbs up: thumb above MCP, other fingers below MCP
    if thumb_tip < thumb_mcp and all(
        f > thumb_mcp for f in [index_tip, middle_tip, ring_tip, pinky_tip]
    ):
        return "thumbs"

    # Peace sign
    if (
        index_tip < thumb_mcp
        and middle_tip < thumb_mcp
        and ring_tip > thumb_mcp
        and pinky_tip > thumb_mcp
    ):
        return "peace"

    return None
