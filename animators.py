# animators.py

import os
import time
import random
from colorama import init, Fore, Style
from frames import FrameManager, Frame

init(autoreset=True)

class BaseAnimator:
    def __init__(self, frame_manager, delay=0.4, repeat=1, clear_screen=True):
        if not isinstance(frame_manager, FrameManager):
            raise ValueError("Must pass a FrameManager.")
        self.frame_manager = frame_manager
        self.delay = delay
        self.repeat = repeat
        self.clear_screen = clear_screen

    def _clear_console(self):
        if self.clear_screen:
            os.system("cls" if os.name == "nt" else "clear")

    def animate(self):
        raise NotImplementedError("Implement in subclasses.")

class SpinningCatAnimator(BaseAnimator):
    def animate(self):
        if len(self.frame_manager) == 0:
            print("No frames to animate.")
            return
        for cycle in range(self.repeat):
            for idx, frame in enumerate(self.frame_manager.get_frames()):
                self._clear_console()
                cycle_str = Fore.YELLOW + f"Cycle {cycle+1}/{self.repeat}"
                frame_str = Fore.CYAN + f"Frame {idx+1}/{len(self.frame_manager)}"
                print(f"{cycle_str}, {frame_str}\n")
                print(Fore.GREEN + str(frame))
                time.sleep(self.delay)

class FancySpinningCatAnimator(SpinningCatAnimator):
    def __init__(self, frame_manager, delay=0.4, repeat=1, clear_screen=True, facts=None):
        super().__init__(frame_manager, delay, repeat, clear_screen)
        self.facts = facts if facts else []

    def _show_fact(self):
        if not self.facts:
            return
        fact = random.choice(self.facts)
        print(Fore.MAGENTA + f"Cat Fact: {fact}\n")

    def animate(self):
        if len(self.frame_manager) == 0:
            print("No frames to animate.")
            return
        for cycle in range(self.repeat):
            for idx, frame in enumerate(self.frame_manager.get_frames()):
                self._clear_console()
                c_str = Fore.YELLOW + f"Fancy Cycle {cycle+1}/{self.repeat}"
                f_str = Fore.CYAN + f"Frame {idx+1}/{len(self.frame_manager)}"
                print(f"{c_str}, {f_str}\n")
                print(Fore.GREEN + str(frame))
                time.sleep(self.delay)
                self._show_fact()
                time.sleep(self.delay / 2)

class ShakySpinningCatAnimator(SpinningCatAnimator):
    def __init__(self, frame_manager, delay=0.4, repeat=1, clear_screen=True, amplitude=2):
        super().__init__(frame_manager, delay, repeat, clear_screen)
        self.amplitude = amplitude

    def _shake_lines(self, text):
        lines = text.split("\n")
        new_lines = []
        for ln in lines:
            offset = random.randint(-self.amplitude, self.amplitude)
            if offset > 0:
                new_lines.append(" " * offset + ln)
            else:
                new_lines.append(ln)
        return "\n".join(new_lines)

    def animate(self):
        if len(self.frame_manager) == 0:
            print("No frames to animate.")
            return
        for cycle in range(self.repeat):
            for idx, frame in enumerate(self.frame_manager.get_frames()):
                self._clear_console()
                c_str = Fore.YELLOW + f"Shaky Cycle {cycle+1}/{self.repeat}"
                f_str = Fore.CYAN + f"Frame {idx+1}/{len(self.frame_manager)}"
                print(f"{c_str}, {f_str}\n")
                shaken = self._shake_lines(frame.ascii_art)
                print(Fore.GREEN + shaken)
                time.sleep(self.delay)
