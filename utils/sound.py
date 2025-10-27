import os

import pygame

SOUND_DIR = "assets/sounds"

pygame.mixer.init()


def get_sound_path(name):
    sound_path = os.path.join(SOUND_DIR, f"{name}.mp3")
    if os.path.exists(sound_path):
        return sound_path
    return None


def play_sound(name):
    path = get_sound_path(name)
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
