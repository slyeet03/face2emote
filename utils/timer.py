import time
from collections import deque

prev_time = time.time()
fps_queue = deque(maxlen=30)
fps = 0.0


def update_fps():
    global prev_time, fps
    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time

    if dt > 0:
        current_fps = 1.0 / dt
        fps_queue.append(current_fps)
        fps = sum(fps_queue) / len(fps_queue)
    return fps
