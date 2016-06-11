import numpy as np


# Single Tiling implementation with equidistant spacing
class SingleTiling:
    # Dimensions is a list containing tuples of the min and max values of each dimension
    def __init__(self, dimensions, num_tiles):
        self.dimensions = dimensions
        self.num_tiles = num_tiles
        self.tiles = np.zeros(len(self.dimensions) ** self.num_tiles)
        self.tile_boundaries = self.__set_tile_boundaries()

    def __set_tile_boundaries(self):
        tile_boundaries = []

        for dimension in self.dimensions:
            tile_range = dimension[1] - dimension[0]
            split = tile_range/float(self.num_tiles)
            tile_boundaries.append([dimension[0] + (i + 1) * split for i in range(self.num_tiles - 1)])

        return tile_boundaries

    def get_value(self, observation):
        return self.tiles[self.__convert_base10(self.__get_tile(observation))]

    def __get_tile(self, observation):
        tile = []

        for i, obs in enumerate(observation):
            for j, space in enumerate(self.tile_boundaries[i]):
                if obs < space:
                    tile.append(j)
                    break

                if j == self.num_tiles - 1:
                    tile.append(j+1)

        return tile

    def __convert_base10(self, tile):
        return sum([val * self.num_tiles ** (len(tile) - (i + 1)) for i, val in enumerate(tile)])



