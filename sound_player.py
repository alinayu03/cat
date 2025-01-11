# sound_player.py

import threading
from playsound import playsound

class SoundPlayer:
    def __init__(self, mp3_path=None, enabled=True):
        self.enabled = enabled
        self.mp3_path = mp3_path

    def play_sound_blocking(self):
        if not self.enabled or not self.mp3_path:
            return
        playsound(self.mp3_path)

    def play_sound_in_background(self):
        if not self.enabled or not self.mp3_path:
            return
        thread = threading.Thread(
            target=playsound, 
            args=(self.mp3_path,),
            daemon=True
        )
        thread.start()
