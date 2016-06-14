import numpy as np


# Single Tiling implementation with equidistant spacing
class SingleTiling:
    # Dimensions is a list containing tuples of the min and max values of each dimension
    def __init__(self, dimensions, num_tiles):
        self.dimensions = dimensions
        self.dimensions_n = len(self.dimensions.low)
        self.num_tiles = num_tiles
        self.tiles = np.zeros(self.dimensions_n ** self.num_tiles)
        self.tile_boundaries = self.__set_tile_boundaries()

    def __set_tile_boundaries(self):
        tile_boundaries = []

        for dim in range(self.dimensions_n):
            #If np.inf then use range of +-1 (CartPole)
            if self.dimensions.high[dim] == np.inf:
                tile_range = 2
                split = tile_range / float(self.num_tiles)
                tile_boundaries.append([-1 + (i + 1) * split for i in range(self.num_tiles - 1)])

            else:
                tile_range = self.dimensions.high[dim] - self.dimensions.low[dim]
                split = tile_range/float(self.num_tiles)
                tile_boundaries.append([self.dimensions.low[dim] + (i + 1) * split for i in range(self.num_tiles - 1)])

        return tile_boundaries

    def get_value(self, observation):
        return self.tiles[self.convert_base10(self.get_tile(observation))]

    def get_tile(self, observation):
        tile = []

        for i, obs in enumerate(observation):
            for j, space in enumerate(self.tile_boundaries[i]):
                if obs < space:
                    tile.append(j)
                    break

                # Has to be minus to otherwise j doesn't reach
                if j == self.num_tiles - 2:
                    tile.append(j+1)

        return tile

    def convert_base10(self, tile):
        return sum([val * self.num_tiles ** (len(tile) - (i + 1)) for i, val in enumerate(tile)])



