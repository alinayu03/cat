# frames.py

import random

class Frame:
    def __init__(self, ascii_art, name=None):
        self.ascii_art = ascii_art
        self.name = name

    def __str__(self):
        return self.ascii_art

class FrameManager:
    def __init__(self):
        self.frames = []

    def add_frame(self, frame):
        if not isinstance(frame, Frame):
            raise TypeError("Only Frame instances allowed.")
        self.frames.append(frame)

    def load_frames(self, items):
        for item in items:
            if isinstance(item, tuple):
                art, nm = item
                self.add_frame(Frame(art, nm))
            else:
                self.add_frame(Frame(item))

    def get_frames(self):
        return self.frames

    def __len__(self):
        return len(self.frames)

class RandomFrameTransformer:
    def __init__(self, flip_chance=0.2, mutate_chance=0.2):
        self.flip_chance = flip_chance
        self.mutate_chance = mutate_chance

    def _flip_text(self, text):
        lines = text.split("\n")
        return "\n".join(lines[::-1])

    def _insert_random_string(self, text):
        lines = text.split("\n")
        insertables = [" ~^~ ", " O.O ", " x_x ", " <3 ", " ^^ ", " *.* "]
        if lines:
            idx = random.randint(0, len(lines)-1)
            extra = random.choice(insertables)
            lines[idx] += extra
        return "\n".join(lines)

    def transform_frames(self, frame_manager):
        for i, f in enumerate(frame_manager.get_frames()):
            transformed = f.ascii_art
            do_flip = random.random() < self.flip_chance
            do_mutate = random.random() < self.mutate_chance
            if do_flip:
                transformed = self._flip_text(transformed)
            if do_mutate:
                transformed = self._insert_random_string(transformed)
            frame_manager.frames[i] = Frame(transformed, f.name)
