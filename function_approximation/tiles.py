import numpy as np


# Single Tiling implementation with equidistant spacing
class SingleTiling:
    # Dimensions is a list containing tuples of the min and max values of each dimension
    def __init__(self, dimensions, num_tiles, resizeable=False):
        self.dimensions = dimensions
        self.dimensions_n = len(self.dimensions.low)
        self.num_tiles = num_tiles

        self.resizeable = resizeable
        self.resize_count = 500
        self.resize_rate = 0.01

        self.tiles = np.zeros(self.dimensions_n ** self.num_tiles)

        self.tile_boundaries = self.__set_tile_boundaries()
        self.tile_hits = self.__set_tile_hits()

    def __set_tile_hits(self):
        if self.resizeable:
            return [np.zeros(self.num_tiles) for _ in range(self.dimensions_n)]

        return None

    def __set_tile_boundaries(self):
        tile_boundaries = []

        for dim in range(self.dimensions_n):
            #If np.inf then use range of +-1 (CartPole)
            if self.dimensions.high[dim] == np.inf:
                split = self.__get_split(dim)
                tile_boundaries.append([-1 + (i + 1) * split for i in range(self.num_tiles - 1)])

            else:
                split = self.__get_split(dim)
                tile_boundaries.append([self.dimensions.low[dim] + (i + 1) * split for i in range(self.num_tiles - 1)])

        return tile_boundaries

    def get_value(self, observation):
        return self.tiles[self.__convert_base10(self.__get_tile(observation))]

    def __get_split(self, obv_ind):
        if self.dimensions.high[obv_ind] == np.inf:
            return 2 / float(self.num_tiles)

        else:
            return (self.dimensions.high[obv_ind] - self.dimensions.low[obv_ind]) / float(self.num_tiles)

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

    def to_array(self, observation):
        return self.__convert_base10(self.__get_tile(observation))
