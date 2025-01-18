class RotatingCatAnimator(BaseAnimator):
    def animate(self):
        if not self.frame_manager:
            return
        cycle = 0
        while cycle < self.repeat:
            idx = 0
            while idx < len(self.frame_manager):
                self._clear_console()
                print(Fore.YELLOW + f"Cycle {cycle + 1}/{self.repeat}, " + 
                      Fore.CYAN + f"Frame {idx + 1}/{len(self.frame_manager)}\n")
                print(Fore.GREEN + str(self.frame_manager.get_frames()[idx]))
                time.sleep(self.delay)
                idx += 1
            cycle += 1
