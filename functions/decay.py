
class DecayBase(object):
    def __init__(self, value, decay, minimum):
        self.minimum = minimum
        self.value = value
        self.decay = decay

        self.can_update = True

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        if v < 0:
            raise ValueError("Decay value cannot be less than 0")

        if v < self.minimum:
            v = self.minimum

        self._value = v

    @property
    def decay(self):
        return self._decay

    @decay.setter
    def decay(self, d):
        if d < 0:
            raise ValueError("Decay rate cannot be less than 0")

        self._decay = d

    @property
    def minimum(self):
        return self._minimum

    @minimum.setter
    def minimum(self, m):
        if m < 0:
            raise ValueError("Decay minimum cannot be less than 0")

        self._minimum = m

    def update(self):
        raise NotImplementedError

    def stop(self):
        self.can_update = False


class FixedDecay(DecayBase):
    def __init__(self, value=1, decay=0.95, minimum=0):
        DecayBase.__init__(self, value, decay, minimum)

    def update(self):
        if not self.can_update:
            return

        self.value *= self.decay

    def __str__(self):
        return "FixedDecay value: {0}, decay: {1}, min: {2}".format(self.value, self.decay, self.minimum)


class EpisodicDecay(DecayBase):
    def __init__(self, decay=.99, minimum=0):
        DecayBase.__init__(self, 1, decay, minimum)
        self.episode_number = 0

    @property
    def episode_number(self):
        return self._episode_number

    @episode_number.setter
    def episode_number(self, en):
        if en < 0:
            raise ValueError("Episode Number must be greater than or equal to 0")

        self._episode_number = en

    def update(self):
        if not self.can_update:
            return

        self.episode_number += 1
        self.value = 1 / self.episode_number ** self.decay

    def __str__(self):
        return "EpisodeNumber value: {0}, decay: {1}, min: {2}, ep num: {3}".format(self.value, self.decay, self.minimum, self.episode_number)
