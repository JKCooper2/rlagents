class FixedDecay:
    def __init__(self, value, decay, minimum=0):
        self.value = value
        self.decay = decay
        self.minimum = minimum

        self.allow_updates = True

    def update(self):
        if not self.allow_updates:
            return

        self.value *= self.decay

        if self.value < self.minimum:
            self.value = self.minimum

    def stop(self):
        self.allow_updates = False


class EpisodeNumber:
    def __init__(self, decay=1, minimum=0):
        self.value = 1
        self.decay = decay
        self.minimum = minimum

        self.episode_number = 0

        self.allow_updates = True

    def update(self):
        if not self.allow_updates:
            return

        self.episode_number += 1
        self.value = max(1 / self.episode_number ** self.decay, self.minimum)

    def stop(self):
        self.allow_updates = False
