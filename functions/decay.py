
class DecayBase(object):
    def __init__(self, value, decay, minimum):
        self.minimum = minimum
        self.value = value
        self.decay = decay

        self.initial_value = self.value

        self.can_update = True

    def export(self):
        raise NotImplementedError

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
    def __init__(self, value=1, decay=1, minimum=0):
        DecayBase.__init__(self, value, decay, minimum)

    def export(self):
        return {"Type": "Fixed Decay",
                "Value": self.initial_value,
                "Decay": self.decay,
                "Minimum": self.minimum}

    def update(self):
        if not self.can_update:
            return

        self.value *= self.decay


class EpisodicDecay(DecayBase):
    def __init__(self, decay=.99, minimum=0):
        DecayBase.__init__(self, 1, decay, minimum)
        self.episode_number = 0

    def export(self):
        return {"Type": "Episodic Decay",
                "Value": self.initial_value,
                "Decay": self.decay,
                "Minimum": self.minimum}

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
