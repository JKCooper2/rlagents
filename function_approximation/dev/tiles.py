import numpy as np


# Single Tiling implementation with equidistant spacing
class SingleTiling:
    # Dimensions is a list containing tuples of the min and max values of each dimension
    def __init__(self, dimensions, num_tiles, update=False):
        self.dimensions = dimensions
        self.dimensions_n = len(self.dimensions.low)
        self.num_tiles = num_tiles
        self.update = update
        self.update_count = 20

        self.tiles = np.zeros(self.dimensions_n ** self.num_tiles)

        self.tile_boundaries = self.__set_tile_boundaries()
        self.tile_hits = self.__set_tile_hits()

    def __set_tile_hits(self):
        if self.update:
            return [np.zeros(self.num_tiles) for _ in range(self.dimensions_n)]

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
            return self.dimensions.high[obv_ind] - self.dimensions.low[obv_ind] / float(self.num_tiles)

    def __update_tile_hits(self, tile):
        for i, obv in enumerate(tile):
            self.tile_hits[i][obv] += 1

        for i in range(len(self.tile_hits)):
            for j in range(len(self.tile_hits[i])):
                if self.tile_hits[i][j] > self.update_count:
                    side_choice = np.random.uniform() > 0.5  # Higher end or lower end to update

                    side = j

                    if j > len(self.tile_boundaries[i]) - 1 and side_choice == 1:
                        pass


                    side = max(0, j-1) if np.random.uniform() < 0.5 else min(len(self.tile_boundaries[i]) - 1, j)
                    print side, i, j

                    times = -1 if side >= j-1 else 1

                    print i, j, self.tile_boundaries[i][j], side, times, split
                    j = min(len(self.tile_boundaries[i]) - 1, j)



                    split = self.__get_split(i)
                    self.tile_boundaries[i][j] += (0.05 * times * split)

                    self.tile_hits[i][j] = 0


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

        if self.update:
            self.__update_tile_hits(tile)

        return tile

    def __convert_base10(self, tile):
        return sum([val * self.num_tiles ** (len(tile) - (i + 1)) for i, val in enumerate(tile)])

    def to_array(self, observation):
        return self.__convert_base10(self.__get_tile(observation))
