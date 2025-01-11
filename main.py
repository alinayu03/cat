# main.py

import sys
from frames import FrameManager, RandomFrameTransformer
from animators import (
    SpinningCatAnimator,
    FancySpinningCatAnimator,
    ShakySpinningCatAnimator
)
from sound_player import SoundPlayer

class AnimationMenu:
    def __init__(self, animators):
        if not isinstance(animators, dict):
            raise ValueError("animators must be dict of name->animator.")
        self.animators = animators

    def _show_menu(self):
        print("\n=== Animation Menu ===")
        i = 1
        for name in self.animators.keys():
            print(f"{i}. {name}")
            i += 1
        print("0. Exit")

    def run(self, sound_player=None):
        while True:
            self._show_menu()
            choice = input("Choose an option: ")
            if choice.isdigit():
                val = int(choice)
            else:
                print("Invalid input.")
                continue
            if val == 0:
                print("Goodbye!")
                break
            keys = list(self.animators.keys())
            if 1 <= val <= len(keys):
                key = keys[val - 1]
                print(f"You chose: {key}")
                if sound_player:
                    # <--- NEW: Start the MP3 on a background thread
                    sound_player.play_sound_in_background()
                # Now run the chosen animation in the main thread
                self.animators[key].animate()
            else:
                print("Invalid choice.")

def build_cat_frames():
    return [
        r"""
 /\_/\  
( o.o ) 
 > ^ <  
""",
        r"""
  |\---/|
 /  o.o  \
 >   ^   <  
""",
        r"""
   .-=-.  
   (o.o) 
    \^/   
""",
        r"""
  |\___/|
 ( o.o ) 
  > ^ <  
""",
        r"""
   /\_/\ 
  ( -.- )
   > ^ <  
""",
        r"""
   |\---/|
  /  -.-  \
  >   ^   <  
""",
        r"""
    .-=-.  
    (-.-) 
     \^/   
""",
        r"""
   |\___/|
  ( -.- )
   > ^ <  
"""
    ]

def main():
    fm = FrameManager()
    fm.load_frames(build_cat_frames())

    transformer = RandomFrameTransformer(flip_chance=0.2, mutate_chance=0.3)
    transformer.transform_frames(fm)

    spinner = SpinningCatAnimator(fm, delay=0.3, repeat=1)
    fancy = FancySpinningCatAnimator(fm, delay=0.4, repeat=1, facts=[
        "A group of cats is called a clowder.",
        "Cats can rotate their ears 180 degrees.",
        "A cat's nose print is unique."
    ])
    shaky = ShakySpinningCatAnimator(fm, delay=0.3, repeat=1, amplitude=3)

    animators = {
        "Basic Spinning Cat": spinner,
        "Fancy Spinning Cat": fancy,
        "Shaky Spinning Cat": shaky
    }

    menu = AnimationMenu(animators)
    sound_player = SoundPlayer(mp3_path="cat.mp3", enabled=True)

    menu.run(sound_player)

if __name__ == "__main__":
    if sys.version_info < (3, 0):
        print("Requires Python 3+.")
        sys.exit(1)
    main()
