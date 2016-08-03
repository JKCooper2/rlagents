
class DecayBase:
    def __init__(self, value, decay, minimum):
        self.value = value
        self.decay = decay
        self.minimum = minimum

        self.allow_updates = True

    def update(self):
        raise NotImplementedError

    def stop(self):
        self.allow_updates = False


class FixedDecay(DecayBase):
    def __init__(self, value=1, decay=0.1, minimum=0):
        DecayBase.__init__(self, value, decay, minimum)

    def update(self):
        if not self.allow_updates:
            return

        self.value *= self.decay

        if self.value < self.minimum:
            self.value = self.minimum

    def __str__(self):
        return "FixedDecay value: {0}, decay: {1}, min: {2}".format(self.value, self.decay, self.minimum)


class EpisodeNumber(DecayBase):
    def __init__(self, decay=1, minimum=0):
        DecayBase.__init__(self, 1, decay, minimum)
        self.episode_number = 0

    def update(self):
        if not self.allow_updates:
            return

        self.episode_number += 1

        self.value = 1 / self.episode_number ** self.decay

        if self.value < self.minimum:
            self.value = self.minimum

    def __str__(self):
        return "EpisodeNumber value: {0}, decay: {1}, min: {2}, ep num: {3}".format(self.value, self.decay, self.minimum, self.episode_number)
