import numpy as np
import copy
from gym.spaces import Discrete, Box, Tuple

"""
Function Approximation takes in a space and apply a function to
a value contained within that space to convert it into another form
"""


class FunctionApproximationBase(object):
    def __init__(self, space=None):
        self.space = space

    @property
    def space_type(self):
        if isinstance(self.space, Discrete):
            return 'D'

        elif isinstance(self.space, Box):
            return 'B'

        elif isinstance(self.space, Tuple):
            return 'T'

    @property
    def num_discrete(self):
        if self.space_type == 'D':
            return self.space.n

        elif self.space_type == 'B':
            return len(self.space.low.flatten())

        elif self.space_type == 'T':
            return len(self.space.spaces)

    @property
    def n_total(self):
        """
        Allows for FA to over-ride the number return while not removing
        accessibility to true space size
        """
        return self.num_discrete

    def convert(self, array):
        """Takes in an action-value array"""
        raise NotImplementedError

    def configure(self, space):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError


class DefaultFA(FunctionApproximationBase):
    def __init__(self, space=None):
        FunctionApproximationBase.__init__(self, space)

    def convert(self, array):
        return array

    def configure(self, space):
        self.space = space

    def export(self):
        return {"Type": "Default"}


class DiscreteMaxFA(FunctionApproximationBase):
    def __init__(self, space=None):
        FunctionApproximationBase.__init__(self, space)

    def convert(self, array):
        action = np.argmax(array)

        if not self.space.contains(action):
            raise ValueError("Action not contained within space")

        return action

    def configure(self, space):
        self.space = space

    def export(self):
        return {"Type": "Discrete Max"}


class ClipFA(FunctionApproximationBase):
    def __init__(self, space=None):
        FunctionApproximationBase.__init__(self, space)

    def convert(self, array):
        if self.space_type == 'B':
            action = np.clip(array, self.space.low, self.space.high)
        else:
            raise TypeError("Can't clip on space type {0}".format(self.space_type))

        return action

    def configure(self, space):
        self.space = space

    def export(self):
        return {"Type": "Clip"}


# Single Tiling implementation with equidistant spacing
class SingleTiling(FunctionApproximationBase):
    # Dimensions is a list containing tuples of the min and max values of each dimension
    def __init__(self, space=None, num_tiles=1, resizeable=False):
        FunctionApproximationBase.__init__(self, space)

        self.num_tiles = num_tiles

        self.resizeable = resizeable
        self.resize_count = 500
        self.resize_rate = 0.01

        self.tiles = None
        self.tile_boundaries = None
        self.tile_hits = None

    def configure(self, space):
        self.space = space

        if self.space_type != 'B':
            raise TypeError("SingleTiling is only valid for box environments")

        self.tiles = np.zeros(self.n_total)
        self.tile_boundaries = self.__set_tile_boundaries()
        self.tile_hits = self.__set_tile_hits()

    @property
    def n_total(self):
        return self.num_discrete ** self.num_tiles

    def __set_tile_hits(self):
        if self.resizeable:
            return [np.zeros(self.num_tiles) for _ in range(self.num_discrete)]

        return None

    def __set_tile_boundaries(self):
        tile_boundaries = []

        for dim in range(self.num_discrete):
            #If np.inf then use range of +-1 (CartPole)
            if self.space.high[dim] == np.inf:
                split = self.__get_split(dim)
                tile_boundaries.append([-1 + (i + 1) * split for i in range(self.num_tiles - 1)])

            else:
                split = self.__get_split(dim)
                tile_boundaries.append([self.space.low[dim] + (i + 1) * split for i in range(self.num_tiles - 1)])

        return tile_boundaries

    def get_value(self, observation):
        return self.tiles[self.__convert_base10(self.__get_tile(observation))]

    def __get_split(self, obv_ind):
        if self.space.high[obv_ind] == np.inf:
            return 2 / float(self.num_tiles)

        else:
            return (self.space.high[obv_ind] - self.space.low[obv_ind]) / float(self.num_tiles)

    def __update_tile_hits(self, tile):
        for i, obv in enumerate(tile):
            self.tile_hits[i][obv] += 1

            if self.tile_hits[i][obv] >= self.resize_count:
                # 0 is lower end, 1 is upper end
                side_choice = int(np.random.uniform() > 0.5)

                change = self.resize_rate * self.__get_split(i)

                if side_choice == 0 and obv > 0:
                    self.tile_boundaries[i][obv - 1] += change

                if side_choice == 1 and obv < len(self.tile_boundaries[i]):
                    self.tile_boundaries[i][obv] -= change

                self.tile_hits[i][obv] = 0

    def __get_tile(self, observation):
        tile = []

        for i, obs in enumerate(observation):
            for j, space in enumerate(self.tile_boundaries[i]):
                if obs < space:
                    tile.append(j)
                    break

                # Has to be minus 2 to otherwise j doesn't reach
                if j == self.num_tiles - 2:
                    tile.append(j+1)

        if self.resizeable:
            self.__update_tile_hits(tile)

        return tile

    def __convert_base10(self, tile):
        return sum([val * self.num_tiles ** (len(tile) - (i + 1)) for i, val in enumerate(tile)])

    def convert(self, observation):
        results = self.__convert_base10(self.__get_tile(observation))
        return results

    def export(self):
        return {"Type": "Single Tiling",
                "Num Tiles": self.num_tiles,
                "Resizeable": self.resizeable}
